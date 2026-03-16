"""
Synthetic Data Self-Improvement Loop
======================================

Generates hard examples by mining low-confidence model outputs,
filters them through the SelfCritiqueModule, and serialises
accepted samples as new SafeTensors training shards.

Usage during training:
    generator = SyntheticDataGenerator(config)
    new_shard = generator.run_epoch_end(
        model_fn=model.apply,
        params=params,
        state=state,
        rng=rng,
        seed_batches=batches,
    )
    # new_shard is a Path to the freshly-written .safetensors file
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generates synthetic hard-example training data by:
    
    1. **Hard-example mining**: Run the model on seed data, keep samples
       where confidence < difficulty_threshold (model finds these hard).
    2. **Critique filtering**: Run SelfCritiqueModule on hard examples,
       keep only samples where critique improves quality by at least
       quality_improvement_min.
    3. **Shard serialisation**: Write filtered examples as a new
       `.safetensors` shard that can be added to the next epoch.
    
    All operations are pure numpy/jax (no Haiku state required) —
    the caller passes `model_fn(params, state, rng, inputs)`.
    """

    def __init__(
        self,
        config: Any,
        output_dir: Optional[str] = None,
    ):
        """
        Args:
            config: AGIConfig instance
            output_dir: Override for shard output directory
        """
        self.difficulty_threshold = getattr(
            config, "synthetic_data_difficulty_threshold", 0.6
        )
        self.batch_multiplier = getattr(
            config, "synthetic_data_batch_multiplier", 0.2
        )
        self.quality_improvement_min = getattr(
            config, "synthetic_data_quality_improvement_min", 0.1
        )
        self.output_dir = Path(
            output_dir or getattr(config, "synthetic_data_output_dir", "synthetic_shards")
        )
        self.vocab_size = getattr(config, "vocab_size", 8000)
        self.seq_length = getattr(config, "max_seq_length", 2048)
        self._shard_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_hard_examples(
        self,
        model_fn: Callable,
        params: Any,
        state: Any,
        rng: jnp.ndarray,
        seed_batches: List[Dict[str, jnp.ndarray]],
        max_examples: Optional[int] = None,
    ) -> List[Dict[str, jnp.ndarray]]:
        """Run model on seed data, keep low-confidence examples.
        
        Args:
            model_fn: ``model.apply(params, state, rng, inputs)``
            params: Model parameters
            state: Model state
            rng: JAX PRNG key
            seed_batches: List of input batches
            max_examples: Cap the number of hard examples
            
        Returns:
            List of batch dicts that the model found hard
        """
        hard_examples: List[Dict[str, jnp.ndarray]] = []
        
        for batch in seed_batches:
            rng, step_rng = jax.random.split(rng)
            
            try:
                inputs = {"text": batch.get("input_ids", batch.get("text"))}
                output, _ = model_fn(params, state, step_rng, inputs)
                
                confidence = output.get("confidence", None)
                if confidence is None:
                    # Fall back to logit entropy as proxy for confidence
                    logits = output.get("logits", None)
                    if logits is not None:
                        probs = jax.nn.softmax(logits, axis=-1)
                        entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
                        max_entropy = jnp.log(jnp.array(self.vocab_size, dtype=jnp.float32))
                        confidence = 1.0 - (entropy.mean() / max_entropy)
                    else:
                        confidence = jnp.array(1.0)
                
                conf_value = float(jnp.mean(confidence))
                
                if conf_value < self.difficulty_threshold:
                    hard_examples.append(batch)
                    
            except Exception as e:
                logger.debug(f"Skipping batch in hard-example mining: {e}")
                continue
            
            if max_examples and len(hard_examples) >= max_examples:
                break
        
        logger.info(
            f"Hard-example mining: {len(hard_examples)} hard examples "
            f"from {len(seed_batches)} seed batches "
            f"(threshold={self.difficulty_threshold})"
        )
        return hard_examples

    def filter_with_critique(
        self,
        examples: List[Dict[str, jnp.ndarray]],
        critique_fn: Callable,
        rng: jnp.ndarray,
    ) -> List[Dict[str, jnp.ndarray]]:
        """Filter examples through critique quality improvement check.
        
        Runs a lightweight quality check on each example and keeps only
        those where the critique-revised output improves by at least
        ``quality_improvement_min``.
        
        Args:
            examples: Hard examples to filter
            critique_fn: Callable that takes hidden_state → dict with
                        ``quality_scores`` list and ``final_quality``
            rng: JAX PRNG key
            
        Returns:
            Filtered list of examples that show quality improvement
        """
        filtered: List[Dict[str, jnp.ndarray]] = []
        
        for batch in examples:
            rng, step_rng = jax.random.split(rng)
            
            try:
                # Use input_ids as a proxy hidden state for critique
                input_ids = batch.get("input_ids", batch.get("text"))
                if input_ids is None:
                    continue
                    
                # Create a synthetic hidden state from embeddings
                # In production, this would use the model's actual hidden output
                batch_size = input_ids.shape[0]
                hidden_dim = min(input_ids.shape[-1], 512)
                hidden_proxy = jax.random.normal(
                    step_rng, (batch_size, hidden_dim)
                )
                
                critique_result = critique_fn(hidden_proxy)
                
                quality_scores = critique_result.get("quality_scores", [])
                if len(quality_scores) >= 2:
                    initial_quality = float(jnp.mean(quality_scores[0]))
                    final_quality = float(jnp.mean(quality_scores[-1]))
                    improvement = final_quality - initial_quality
                    
                    if improvement >= self.quality_improvement_min:
                        filtered.append(batch)
                else:
                    # Single quality score — keep if below threshold (room to improve)
                    final_q = float(jnp.mean(critique_result.get("final_quality", 1.0)))
                    if final_q < self.difficulty_threshold:
                        filtered.append(batch)
                        
            except Exception as e:
                logger.debug(f"Skipping batch in critique filter: {e}")
                continue
        
        logger.info(
            f"Critique filtering: {len(filtered)}/{len(examples)} examples passed "
            f"(min improvement={self.quality_improvement_min})"
        )
        return filtered

    def augment_training_shard(
        self,
        filtered_examples: List[Dict[str, jnp.ndarray]],
        epoch: int = 0,
    ) -> Optional[Path]:
        """Serialise filtered examples as a new .safetensors shard.
        
        Args:
            filtered_examples: Critique-filtered hard examples
            epoch: Current training epoch (for filename)
            
        Returns:
            Path to the new shard file, or None if nothing to write
        """
        if not filtered_examples:
            logger.info("No examples to write — skipping shard creation")
            return None
        
        # Concatenate all examples
        all_input_ids = []
        all_targets = []
        
        for batch in filtered_examples:
            input_ids = batch.get("input_ids", batch.get("text"))
            targets = batch.get("targets", input_ids)
            
            if input_ids is not None:
                all_input_ids.append(np.asarray(input_ids))
                all_targets.append(np.asarray(targets))
        
        if not all_input_ids:
            return None
        
        combined_inputs = np.concatenate(all_input_ids, axis=0)
        combined_targets = np.concatenate(all_targets, axis=0)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write shard
        self._shard_counter += 1
        shard_name = f"synthetic_epoch{epoch}_shard{self._shard_counter:04d}.safetensors"
        shard_path = self.output_dir / shard_name
        
        try:
            from safetensors.numpy import save_file as save_safetensors
            
            save_safetensors(
                {
                    "input_ids": combined_inputs,
                    "targets": combined_targets,
                },
                str(shard_path),
            )
            
            logger.info(
                f"Saved synthetic shard: {shard_path} "
                f"({combined_inputs.shape[0]} examples)"
            )
            return shard_path
            
        except ImportError:
            logger.warning("safetensors not available — saving as .npy instead")
            npy_path = shard_path.with_suffix(".npy")
            np.save(str(npy_path), {
                "input_ids": combined_inputs,
                "targets": combined_targets,
            })
            return npy_path

    def run_epoch_end(
        self,
        model_fn: Callable,
        params: Any,
        state: Any,
        rng: jnp.ndarray,
        seed_batches: List[Dict[str, jnp.ndarray]],
        critique_fn: Optional[Callable] = None,
        epoch: int = 0,
    ) -> Optional[Path]:
        """Full end-of-epoch pipeline: mine → filter → save.
        
        Args:
            model_fn: model.apply callable
            params: Model parameters
            state: Model state
            rng: JAX PRNG key
            seed_batches: Seed data from this epoch
            critique_fn: Optional critique function (SelfCritiqueModule.__call__)
            epoch: Current epoch number
            
        Returns:
            Path to new shard, or None
        """
        t0 = time.time()
        
        # 1. Determine how many batches to mine
        max_hard = max(1, int(len(seed_batches) * self.batch_multiplier))
        
        rng, mine_rng, filter_rng = jax.random.split(rng, 3)
        
        # 2. Mine hard examples
        hard = self.generate_hard_examples(
            model_fn, params, state, mine_rng,
            seed_batches, max_examples=max_hard,
        )
        
        if not hard:
            logger.info("No hard examples found — nothing to generate")
            return None
        
        # 3. Filter with critique (if available)
        if critique_fn is not None:
            filtered = self.filter_with_critique(hard, critique_fn, filter_rng)
        else:
            filtered = hard
        
        # 4. Save shard
        shard_path = self.augment_training_shard(filtered, epoch=epoch)
        
        elapsed = time.time() - t0
        logger.info(f"Synthetic data pipeline took {elapsed:.1f}s")
        
        return shard_path
