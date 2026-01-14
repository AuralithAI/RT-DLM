import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import re
import argparse
from typing import Dict, List, Tuple, Optional, Union, Iterator, Any
import matplotlib.pyplot as plt
import time
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project paths using absolute path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from rtdlm import (
    create_rtdlm_agi, create_agi_optimizer, 
    compute_agi_loss
)
from core.checkpoint_manager import CheckpointManager
from config.agi_config import AGIConfig
from config.model_parallel_config import ModelParallelConfig
from modules.capabilities.advanced_algorithms import (
    TaskMemory, compute_ewc_loss, compute_fisher_information
)
# Model parallelism imports (used when model_parallel=True)
from core.model_parallel import (
    DeviceMesh,
    create_model_parallel_system,
    create_model_parallel_transformer,
)

def cosine_similarity(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compute cosine similarity between two vectors"""
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = jnp.linalg.norm(a_flat) + 1e-8
    norm_b = jnp.linalg.norm(b_flat) + 1e-8
    return jnp.sum(a_flat * b_flat) / (norm_a * norm_b)

class AGITrainer:
    """
    Advanced trainer for RT-DLM AGI.
    
    Accepts pre-tokenized tensor data from the external data pipeline.
    Training data should be prepared using Auralith-Data-Pipeline.
    
    Supports:
    - Standard single-device training
    - Multi-device data parallelism (distributed_training=True)
    - Model parallelism for very large models (model_parallel=True)
    """
    
    def __init__(self, config: AGIConfig):
        self.config = config
        self.rng = jax.random.PRNGKey(42)
        
        # Model parallelism setup (for very large models)
        self.device_mesh = None
        self.mp_config = None
        if config.model_parallel:
            self.device_mesh, self.mp_config = create_model_parallel_system(config)
            logger.info(f"Model parallelism enabled: {self.mp_config.tensor_parallel_size} devices")
        
        # Initialize model (standard or model-parallel)
        if config.model_parallel and self.device_mesh is not None:
            # Use model parallel transformer for very large models
            self.model = create_model_parallel_transformer(config, self.device_mesh)
            logger.info("Using model-parallel transformer architecture")
        else:
            # Standard model
            self.model = create_rtdlm_agi(config)
        
        # Initialize optimizer
        self.optimizer = create_agi_optimizer(config)
        
        # Training state
        self.params = None
        self.opt_state = None
        self.step_count = 0
        
        # Metrics tracking
        self.training_losses = []
        self.validation_losses = []
        self.reasoning_accuracies = []
        self.consciousness_coherence = []
        self.multimodal_alignment = []
        
        # Continual learning state
        self.task_memories: List[TaskMemory] = []
        self.current_task_id = "task_0"
        self.lambda_ewc = 1000.0
        
    def initialize_model(self, sample_batch: Dict[str, jnp.ndarray]):
        """Initialize model parameters from a sample batch."""
        logger.info("Initializing RT-DLM model...")
        
        # Create sample inputs for initialization
        sample_inputs = {
            "text": sample_batch["input_ids"],
        }
        
        # Add multimodal samples if enabled
        if self.config.multimodal_enabled:
            batch_size = sample_batch["input_ids"].shape[0]
            sample_inputs["multimodal_inputs"] = {
                "images": jnp.zeros((batch_size, 224, 224, 3)),
                "audio": jnp.zeros((batch_size, 128, 128)),
            }
        
        # Initialize parameters
        self.rng, init_rng = jax.random.split(self.rng)
        self.params = self.model.init(init_rng, **sample_inputs)
        self.opt_state = self.optimizer.init(self.params)
        
        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        logger.info(f"Model initialized with {param_count:,} parameters")
        
        return self.params, self.opt_state
    
    def consolidate_task(self, task_id: str, data_samples: jnp.ndarray, targets: jnp.ndarray):
        """
        Consolidate knowledge for a completed task using EWC.
        
        Computes Fisher information and stores task memory for future
        regularization to prevent catastrophic forgetting.
        
        Args:
            task_id: Unique identifier for the task
            data_samples: Representative data samples from the task
            targets: Corresponding targets
        """
        if self.params is None:
            raise RuntimeError("Model must be initialized before consolidating tasks")
            
        logger.info(f"Consolidating task {task_id} with EWC...")
        
        # Compute Fisher information for current task
        self.rng, fisher_rng = jax.random.split(self.rng)
        fisher_matrix = compute_fisher_information(
            model_fn=self.model.apply,
            params=self.params,
            data_samples=data_samples,
            targets=targets,
            rng=fisher_rng,
            num_samples=min(100, len(data_samples))
        )
        
        # Create task memory
        task_memory = TaskMemory(
            task_id=task_id,
            params_snapshot=jax.tree_util.tree_map(lambda x: x.copy(), self.params),
            fisher_matrix=fisher_matrix,
            importance_weights=fisher_matrix,
            performance_metrics={"final_loss": float(self.training_losses[-1]) if self.training_losses else 0.0},
            num_samples=len(data_samples)
        )
        
        self.task_memories.append(task_memory)
        logger.info(f"Task {task_id} consolidated. Total task memories: {len(self.task_memories)}")
    
    def compute_ewc_regularization(self, params: Dict) -> jnp.ndarray:
        """
        Compute EWC regularization loss from all previous task memories.
        
        Args:
            params: Current model parameters
            
        Returns:
            EWC regularization loss
        """
        if not self.task_memories:
            return jnp.float32(0.0)
        
        total_ewc_loss = jnp.float32(0.0)
        for memory in self.task_memories:
            ewc_loss = compute_ewc_loss(
                params=params,
                params_star=memory.params_snapshot,
                fisher_matrix=memory.fisher_matrix,
                lambda_ewc=self.lambda_ewc
            )
            total_ewc_loss += ewc_loss
        
        return total_ewc_loss

    def create_batch_from_tensors(
        self, 
        input_ids: jnp.ndarray,
        targets: Optional[jnp.ndarray] = None,
        images: Optional[jnp.ndarray] = None,
        audio: Optional[jnp.ndarray] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Create a training batch from pre-tokenized tensor data.
        
        Data should be prepared using Auralith-Data-Pipeline and loaded
        from SafeTensors shards.
        
        Args:
            input_ids: Pre-tokenized input tensor (batch, seq_len)
            targets: Target tensor for next-token prediction (default: input_ids)
            images: Optional image tensor (batch, H, W, C)
            audio: Optional audio mel spectrogram tensor (batch, time, freq)
        
        Returns:
            Dictionary with input_ids, targets, and optional multimodal_inputs
        """
        batch = {
            "input_ids": input_ids,
            "targets": targets if targets is not None else input_ids,
            "text": input_ids,
        }
        
        batch_size = input_ids.shape[0]
        
        # Add multimodal data if provided
        if self.config.multimodal_enabled and (images is not None or audio is not None):
            self.rng, *modal_rngs = jax.random.split(self.rng, 3)
            
            # Use provided images or generate synthetic
            if images is not None:
                batch_images = jnp.array(images[:batch_size], dtype=jnp.float32)
            else:
                batch_images = jax.random.normal(modal_rngs[0], (batch_size, 224, 224, 3)) * 0.1
            
            # Use provided audio or generate synthetic
            if audio is not None:
                mel_spectrograms = jnp.array(audio[:batch_size], dtype=jnp.float32)
            else:
                mel_spectrograms = jax.random.normal(modal_rngs[1], (batch_size, 128, 128)) * 0.1
            
            batch["image"] = batch_images
            batch["audio"] = mel_spectrograms
            batch["multimodal_inputs"] = {
                "images": batch_images,
                "audio": mel_spectrograms,
            }
        
        return batch
    
    def create_reasoning_batch(self, batch_size: int = 8) -> Dict[str, jnp.ndarray]:
        """
        Create a synthetic reasoning task batch for evaluation.
        
        Note: For actual training, use pre-tokenized data from the data pipeline.
        This is only for reasoning capability evaluation.
        """
        # Generate random token sequences for reasoning evaluation
        self.rng, rng = jax.random.split(self.rng)
        input_ids = jax.random.randint(
            rng, 
            (batch_size, self.config.max_seq_length), 
            0, 
            self.config.vocab_size
        )
        return self.create_batch_from_tensors(input_ids)
    
    @jax.jit
    def train_step(self, params, opt_state, batch, rng):
        """Single training step with comprehensive loss and NaN handling"""
        
        def loss_fn(params, batch, rng):
            # Forward pass
            model_output = self.model.apply(
                params, rng,
                inputs={"text": batch["input_ids"]},
                multimodal_inputs=batch.get("multimodal_inputs"),
                return_reasoning=True
            )
            
            # Compute comprehensive loss
            loss = compute_agi_loss(
                model_output["logits"],
                batch["targets"],
                aux_outputs=model_output,
                config=self.config
            )
            
            return loss, model_output
        
        # Compute gradients
        (loss, model_output), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params, batch, rng)
        
        # NaN check for loss - replace with zero if NaN detected
        loss = jax.lax.cond(
            jnp.isnan(loss),
            lambda _: jnp.float32(0.0),
            lambda l: l,
            loss
        )
        
        # NaN check for gradients - zero out NaN gradients
        def zero_nan_grads(g):
            return jax.tree_util.tree_map(
                lambda x: jnp.where(jnp.isnan(x), jnp.zeros_like(x), x), 
                g
            )
        grads = zero_nan_grads(grads)
        
        # Update parameters
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss, model_output
    
    def train_step_with_ewc(self, params, opt_state, batch, rng, 
                            task_memories: List[TaskMemory], lambda_ewc: float = 1000.0):
        """
        Training step with EWC regularization for continual learning.
        
        Adds EWC loss to prevent catastrophic forgetting of previous tasks.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            batch: Training batch
            rng: Random key
            task_memories: List of previous task memories
            lambda_ewc: EWC regularization strength
            
        Returns:
            Updated params, optimizer state, total loss, EWC loss component
        """
        def loss_fn_with_ewc(params, batch, rng):
            # Forward pass
            model_output = self.model.apply(
                params, rng,
                inputs={"text": batch["input_ids"]},
                multimodal_inputs=batch.get("multimodal_inputs"),
                return_reasoning=True
            )
            
            # Compute base AGI loss
            base_loss = compute_agi_loss(
                model_output["logits"],
                batch["targets"],
                aux_outputs=model_output,
                config=self.config
            )
            
            # Compute EWC regularization loss
            ewc_loss = jnp.float32(0.0)
            for memory in task_memories:
                ewc_loss += compute_ewc_loss(
                    params=params,
                    params_star=memory.params_snapshot,
                    fisher_matrix=memory.fisher_matrix,
                    lambda_ewc=lambda_ewc
                )
            
            total_loss = base_loss + ewc_loss
            
            return total_loss, (model_output, base_loss, ewc_loss)
        
        # Compute gradients
        (total_loss, (_, base_loss, ewc_loss)), grads = jax.value_and_grad(
            loss_fn_with_ewc, has_aux=True
        )(params, batch, rng)
        
        # NaN check for loss
        total_loss = jax.lax.cond(
            jnp.isnan(total_loss),
            lambda _: jnp.float32(0.0),
            lambda l: l,
            total_loss
        )
        
        # NaN check for gradients
        def zero_nan_grads(g):
            return jax.tree_util.tree_map(
                lambda x: jnp.where(jnp.isnan(x), jnp.zeros_like(x), x), 
                g
            )
        grads = zero_nan_grads(grads)
        
        # Update parameters
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, total_loss, base_loss, ewc_loss

    def evaluate_model(self, eval_batches: List[Dict[str, jnp.ndarray]], num_samples: int = 100):
        """
        Comprehensive model evaluation on pre-tokenized tensor data.
        
        Args:
            eval_batches: List of evaluation batches, each containing input_ids and targets
            num_samples: Maximum number of samples to evaluate
        """
        logger.info("Evaluating model...")
        
        eval_losses = []
        reasoning_scores = []
        consciousness_scores = []
        
        # Regular evaluation
        samples_evaluated = 0
        for batch in eval_batches:
            if samples_evaluated >= num_samples:
                break
            
            # Forward pass
            self.rng, eval_rng = jax.random.split(self.rng)
            model_output = self.model.apply(
                self.params, eval_rng,
                inputs={"text": batch["input_ids"]},
                return_reasoning=True
            )
            
            # Compute loss
            loss = compute_agi_loss(
                model_output["logits"],
                batch["targets"],
                aux_outputs=model_output,
                config=self.config
            )
            
            eval_losses.append(float(loss))
            samples_evaluated += batch["input_ids"].shape[0]
            
            # Evaluate reasoning quality
            if "reasoning_chain" in model_output:
                reasoning_score = self.evaluate_reasoning_quality(
                    model_output["reasoning_chain"]
                )
                reasoning_scores.append(reasoning_score)
            
            # Evaluate consciousness coherence
            if "consciousness" in model_output:
                consciousness_score = self.evaluate_consciousness_coherence(
                    model_output["consciousness"]
                )
                consciousness_scores.append(consciousness_score)
        
        # Reasoning-specific evaluation
        reasoning_batch = self.create_reasoning_batch(self.config.batch_size)
        self.rng, reasoning_rng = jax.random.split(self.rng)
        
        reasoning_output = self.model.apply(
            self.params, reasoning_rng,
            inputs={"text": reasoning_batch["input_ids"]},
            return_reasoning=True
        )
        
        reasoning_accuracy = self.evaluate_reasoning_accuracy(reasoning_output)
        
        metrics = {
            "eval_loss": np.mean(eval_losses) if eval_losses else float('inf'),
            "reasoning_accuracy": reasoning_accuracy,
            "reasoning_quality": np.mean(reasoning_scores) if reasoning_scores else 0.0,
            "consciousness_coherence": np.mean(consciousness_scores) if consciousness_scores else 0.0,
        }
        
        return metrics
    
    def evaluate_reasoning_quality(self, reasoning_chain, ground_truth: Optional[str] = None):
        """Evaluate the quality of reasoning chain with regex parsing"""
        if not reasoning_chain:
            return 0.0
        
        # Check for consistency between steps
        consistency_score = 0.0
        for i in range(len(reasoning_chain) - 1):
            step_similarity = cosine_similarity(
                reasoning_chain[i],
                reasoning_chain[i+1]
            )
            consistency_score += float(step_similarity)
        
        base_score = consistency_score / max(1, len(reasoning_chain) - 1)
        
        # If ground truth provided, compute accuracy via regex parsing
        if ground_truth is not None:
            # Parse step-by-step reasoning patterns
            # Matches patterns like "Step 1: ...", "step 2: ...", "Answer: ..."
            step_pattern = r'[Ss]tep\s*\d+:\s*(.+)(?=[Ss]tep\s*\d+:|[Aa]nswer:|$)'
            answer_pattern = r'[Aa]nswer:\s*([^.]+)'
            
            # Extract steps and answer from reasoning output (if available as text)
            # This works with string representations of the reasoning chain
            reasoning_text = str(reasoning_chain)
            
            steps = re.findall(step_pattern, reasoning_text, re.DOTALL)
            answers = re.findall(answer_pattern, reasoning_text)
            
            # Compute accuracy based on matching
            if answers and ground_truth:
                # Simple string matching for answer accuracy
                answer_text = answers[-1].strip().lower()
                gt_text = ground_truth.strip().lower()
                
                # Check if answer contains ground truth or vice versa
                if gt_text in answer_text or answer_text in gt_text:
                    return min(1.0, base_score + 0.5)  # Boost score for correct answer
            
            # Reward having multiple reasoning steps
            step_bonus = min(0.3, len(steps) * 0.1)
            return min(1.0, base_score + step_bonus)
        
        return base_score
    
    def evaluate_consciousness_coherence(self, consciousness_signals):
        """Evaluate consciousness simulation coherence"""
        if not consciousness_signals:
            return 0.0
        
        coherence = 0.0
        count = 0
        
        # Check alignment between self-awareness and introspection
        if "self_awareness" in consciousness_signals and "introspection" in consciousness_signals:
            alignment = cosine_similarity(
                consciousness_signals["self_awareness"],
                consciousness_signals["introspection"].mean(axis=1)
            )
            coherence += float(alignment)
            count += 1
        
        # Check goal consistency
        if "autonomous_goals" in consciousness_signals:
            goal_consistency = jnp.std(consciousness_signals["autonomous_goals"])
            coherence += float(1.0 / (1.0 + goal_consistency))  # Lower std = higher consistency
            count += 1
        
        return coherence / max(1, count)
    
    def evaluate_reasoning_accuracy(self, reasoning_output):
        """Evaluate accuracy on reasoning tasks"""
        # This is a simplified evaluation
        # In practice, you'd parse the generated text and check mathematical correctness
        
        if "reasoning_chain" in reasoning_output:
            chain_length = len(reasoning_output["reasoning_chain"])
            # Assume longer reasoning chains indicate better reasoning
            return min(1.0, chain_length / self.config.max_reasoning_steps)
        
        return 0.0
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save training checkpoint using SafeTensors (secure format)"""
        checkpoint_manager = CheckpointManager(
            checkpoint_dir="checkpoints",
            model_name="rtdlm_agi",
            keep_last_n=5  # Keep last 5 checkpoints
        )
        
        checkpoint_manager.save_checkpoint(
            params=self.params,
            opt_state=self.opt_state,
            epoch=epoch,
            step_count=self.step_count,
            metrics=metrics,
            config=self.config.to_dict(),
            training_losses=self.training_losses[-100:],  # Last 100 losses
            validation_losses=self.validation_losses
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint from SafeTensors format"""
        checkpoint_manager = CheckpointManager(checkpoint_dir="checkpoints")
        
        checkpoint = checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            load_opt_state=True,
            reference_opt_state=self.opt_state
        )
        
        self.params = checkpoint["params"]
        if checkpoint["opt_state"] is not None:
            self.opt_state = checkpoint["opt_state"]
        self.step_count = checkpoint.get("step_count", 0)
        
        print(f"[INFO] Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint
    
    def resume_from_checkpoint(self, checkpoint_path: str, sample_batch: Dict) -> int:
        """
        Resume training from a checkpoint.
        
        Loads model parameters, optimizer state, and training progress from a 
        previously saved checkpoint. The model doesn't start from zero weights.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.safetensors)
            sample_batch: Sample batch for model initialization (if needed)
            
        Returns:
            The epoch number to resume from (start_epoch)
        """
        print(f"[INFO] Resuming training from checkpoint: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # First, initialize the model structure if not already done
        if self.params is None:
            logger.info("Initializing model structure before loading weights...")
            self.initialize_model(sample_batch)
        
        # Load checkpoint
        checkpoint_manager = CheckpointManager(checkpoint_dir=str(Path(checkpoint_path).parent))
        
        checkpoint = checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            load_opt_state=True,
            reference_opt_state=self.opt_state
        )
        
        # Restore model parameters
        self.params = checkpoint["params"]
        
        # Restore optimizer state if available
        if checkpoint["opt_state"] is not None:
            self.opt_state = checkpoint["opt_state"]
            logger.info("Optimizer state restored")
        else:
            logger.info("No optimizer state in checkpoint, using fresh optimizer")
        
        # Restore step count
        self.step_count = checkpoint.get("step_count", 0)
        
        # Restore training history if available
        metadata = checkpoint.get("metadata", {})
        if metadata.get("training_losses"):
            self.training_losses = list(metadata["training_losses"])
            logger.info(f"Restored {len(self.training_losses)} training loss records")
        if metadata.get("validation_losses"):
            self.validation_losses = list(metadata["validation_losses"])
            logger.info(f"Restored {len(self.validation_losses)} validation loss records")
        
        # Calculate resume epoch
        resume_epoch = checkpoint.get("epoch", 0)
        
        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        
        logger.info("Checkpoint loaded successfully")
        logger.info(f"  - Epoch: {resume_epoch}")
        logger.info(f"  - Step count: {self.step_count}")
        logger.info(f"  - Parameters: {param_count:,}")
        
        return resume_epoch
    
    def plot_training_metrics(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(self.training_losses)
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        
        # Validation loss
        if self.validation_losses:
            axes[0, 1].plot(self.validation_losses)
            axes[0, 1].set_title("Validation Loss")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Loss")
        
        # Reasoning accuracy
        if self.reasoning_accuracies:
            axes[1, 0].plot(self.reasoning_accuracies)
            axes[1, 0].set_title("Reasoning Accuracy")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Accuracy")
        
        # Consciousness coherence
        if self.consciousness_coherence:
            axes[1, 1].plot(self.consciousness_coherence)
            axes[1, 1].set_title("Consciousness Coherence")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Coherence Score")
        
        plt.tight_layout()
        plt.savefig("agi_training_metrics.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)  # Clean up figure resources
    
    def _run_epoch(
        self,
        batch_iterator: Iterator[Dict[str, jnp.ndarray]],
        num_batches_estimate: int
    ) -> List[float]:
        """
        Run a single training epoch.
        
        Args:
            batch_iterator: Iterator over training batches
            num_batches_estimate: Estimated number of batches for logging
            
        Returns:
            List of losses for the epoch
        """
        epoch_losses = []
        
        for batch_idx, batch in enumerate(batch_iterator):
            self.rng, train_rng = jax.random.split(self.rng)
            self.params, self.opt_state, loss, _ = self.train_step(
                self.params, self.opt_state, batch, train_rng
            )
            
            if jnp.isnan(loss) or jnp.isinf(loss):
                logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping...")
                continue
            
            epoch_losses.append(float(loss))
            self.training_losses.append(float(loss))
            self.step_count += 1
            
            if batch_idx % 50 == 0:
                logger.info(f"  Batch {batch_idx + 1}/{num_batches_estimate}, Loss: {loss:.4f}")
        
        return epoch_losses
    
    def _run_validation(
        self, 
        val_data: Any,
        use_streaming: bool
    ) -> Dict[str, float]:
        """
        Run validation and return metrics.
        
        Args:
            val_data: Validation data source (List or ShardedDataLoader)
            use_streaming: Whether using streaming data loader
            
        Returns:
            Dictionary of validation metrics
        """
        if use_streaming:
            val_batches: List[Dict[str, jnp.ndarray]] = val_data.get_validation_batches(max_batches=50)
        else:
            val_batches = val_data
            
        return self.evaluate_model(val_batches)
    
    def _check_early_stopping(
        self,
        val_loss: float,
        best_val_loss: float,
        patience_counter: int,
        max_patience: int,
        epoch: int,
        metrics: Dict[str, float]
    ) -> Tuple[float, int, bool]:
        """
        Check early stopping condition and update best model.
        
        Returns:
            Tuple of (new_best_loss, new_patience_counter, should_stop)
        """
        if val_loss < best_val_loss:
            self.save_checkpoint(epoch, metrics)
            return val_loss, 0, False
        
        new_patience = patience_counter + 1
        if new_patience >= max_patience:
            logger.info(f"Early stopping triggered after {new_patience} epochs without improvement")
            return best_val_loss, new_patience, True
        
        return best_val_loss, new_patience, False
    
    def _prepare_training(
        self,
        train_data: Any,
        resume_checkpoint: Optional[str]
    ) -> Tuple[int, Dict[str, jnp.ndarray], int, bool]:
        """
        Prepare training: initialize model or resume from checkpoint.
        
        Returns:
            Tuple of (start_epoch, sample_batch, num_batches_estimate, use_streaming)
        """
        use_streaming = isinstance(train_data, ShardedDataLoader)
        
        if use_streaming:
            sample_batch = train_data.get_sample_batch()
            num_batches_estimate = train_data.num_batches_per_epoch
        else:
            sample_batch = train_data[0]
            num_batches_estimate = len(train_data)
        
        start_epoch = 0
        if resume_checkpoint:
            start_epoch = self.resume_from_checkpoint(resume_checkpoint, sample_batch)
            logger.info(f"Resuming from epoch {start_epoch + 1}")
        else:
            self.initialize_model(sample_batch)
        
        return start_epoch, sample_batch, num_batches_estimate, use_streaming

    def train(
        self, 
        train_data: Union[List[Dict[str, jnp.ndarray]], "ShardedDataLoader"],
        val_data: Union[List[Dict[str, jnp.ndarray]], "ShardedDataLoader"],
        resume_checkpoint: Optional[str] = None
    ) -> Dict:
        """
        Complete training loop for RT-DLM.
        
        Accepts pre-tokenized tensor data from the external data pipeline
        (Auralith-Data-Pipeline). Supports both in-memory batch lists and
        streaming ShardedDataLoader for large datasets.
        
        Args:
            train_data: Training data - either list of batches or ShardedDataLoader
            val_data: Validation data - either list of batches or ShardedDataLoader
            resume_checkpoint: Optional path to checkpoint to resume from.
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("=" * 60)
        logger.info("RT-DLM Training Started")
        logger.info("=" * 60)
        
        self.config.print_summary()
        
        # Prepare training
        start_epoch, _, num_batches_estimate, use_streaming = self._prepare_training(
            train_data, resume_checkpoint
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 5
        
        for epoch in range(start_epoch, self.config.num_epochs):
            logger.info(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
            epoch_start_time = time.time()
            
            # Run training epoch
            batch_iterator = train_data.iter_epoch() if use_streaming else iter(train_data)
            epoch_losses = self._run_epoch(batch_iterator, num_batches_estimate)
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
            logger.info(f"  Average Loss: {avg_epoch_loss:.4f}")
            logger.info(f"  Epoch Time: {epoch_time:.1f}s")
            
            # Periodic checkpoint
            if epoch % 10 == 0 and epoch > 0:
                self.save_checkpoint(epoch, {"avg_loss": float(avg_epoch_loss)})
                logger.info(f"Periodic checkpoint saved at epoch {epoch}")
            
            # Validation
            should_validate = (epoch % self.config.eval_interval == 0) or (epoch == self.config.num_epochs - 1)
            if should_validate:
                metrics = self._run_validation(val_data, use_streaming)
                
                self.validation_losses.append(metrics["eval_loss"])
                self.reasoning_accuracies.append(metrics["reasoning_accuracy"])
                self.consciousness_coherence.append(metrics["consciousness_coherence"])
                
                logger.info(f"  Validation Loss: {metrics['eval_loss']:.4f}")
                logger.info(f"  Reasoning Accuracy: {metrics['reasoning_accuracy']:.4f}")
                logger.info(f"  Consciousness Coherence: {metrics['consciousness_coherence']:.4f}")
                
                # Check early stopping
                best_val_loss, patience_counter, should_stop = self._check_early_stopping(
                    metrics["eval_loss"], best_val_loss, patience_counter, max_patience, epoch, metrics
                )
                if should_stop:
                    break
        
        logger.info("=" * 60)
        logger.info("RT-DLM Training Completed!")
        logger.info("=" * 60)
        
        self.plot_training_metrics()
        
        return {
            "final_params": self.params,
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "reasoning_accuracies": self.reasoning_accuracies,
            "consciousness_coherence": self.consciousness_coherence,
        }


def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train RT-DLM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch
  python train.py
  
  # Resume training from a checkpoint
  python train.py --resume checkpoints/rtdlm_agi_epoch_10.safetensors
  
  # Train with custom settings
  python train.py --epochs 50 --batch-size 32 --lr 1e-4
  
  # Resume with more epochs
  python train.py --resume checkpoints/rtdlm_agi_epoch_10.safetensors --epochs 100
        """
    )
    
    # Checkpoint resumption
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to checkpoint file to resume training from (.safetensors)"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        help="Total number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    
    # Model configuration
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Model hidden dimension (default: 512)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=12,
        help="Number of transformer layers (default: 12)"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)"
    )
    
    # Data paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to directory containing pre-tokenized SafeTensor shards"
    )
    
    # Output
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)"
    )
    
    return parser.parse_args()


class ShardedDataLoader:
    """
    Memory-efficient data loader for large SafeTensor datasets.
    
    Streams batches from disk instead of loading all data into memory.
    Supports sharding across multiple files for datasets that don't fit in RAM.
    """
    
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int,
        seq_length: int,
        shuffle: bool = True,
        prefetch_shards: int = 2
    ):
        """
        Initialize the sharded data loader.
        
        Args:
            data_dir: Directory containing .safetensors shard files
            batch_size: Number of samples per batch
            seq_length: Maximum sequence length
            shuffle: Whether to shuffle shards each epoch
            prefetch_shards: Number of shards to keep in memory
        """
        from safetensors.numpy import load_file as load_safetensors
        self._load_safetensors = load_safetensors
        
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.prefetch_shards = prefetch_shards
        
        # Discover shard files
        self.shard_files = sorted(self.data_dir.glob("*.safetensors"))
        if not self.shard_files:
            raise FileNotFoundError(f"No .safetensors files found in {data_dir}")
        
        self.num_shards = len(self.shard_files)
        self._rng = np.random.default_rng(42)
        
        # Estimate total samples (scan first shard)
        first_shard = self._load_shard(self.shard_files[0])
        self.samples_per_shard = first_shard["input_ids"].shape[0]
        self.total_samples = self.samples_per_shard * self.num_shards
        
        logger.info(f"DataLoader initialized: {self.num_shards} shards, ~{self.total_samples} samples")
    
    def _load_shard(self, shard_path: Path) -> Dict[str, np.ndarray]:
        """Load a single shard from disk."""
        tensors = self._load_safetensors(str(shard_path))
        return tensors
    
    def _create_batches_from_shard(
        self, 
        shard_data: Dict[str, np.ndarray]
    ) -> List[Dict[str, jnp.ndarray]]:
        """Create batches from a loaded shard."""
        input_ids = shard_data["input_ids"]
        targets = shard_data.get("targets", input_ids)
        
        num_samples = input_ids.shape[0]
        batches = []
        
        # Create batches from this shard
        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            
            batch_input_ids = jnp.array(input_ids[start_idx:end_idx])
            batch_targets = jnp.array(targets[start_idx:end_idx])
            
            # Pad to batch_size if needed (for last batch)
            current_batch_size = batch_input_ids.shape[0]
            if current_batch_size < self.batch_size:
                pad_size = self.batch_size - current_batch_size
                batch_input_ids = jnp.pad(
                    batch_input_ids, 
                    ((0, pad_size), (0, 0)), 
                    constant_values=0
                )
                batch_targets = jnp.pad(
                    batch_targets, 
                    ((0, pad_size), (0, 0)), 
                    constant_values=0
                )
            
            batches.append({
                "input_ids": batch_input_ids,
                "targets": batch_targets,
                "text": batch_input_ids,
            })
        
        return batches
    
    def iter_epoch(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """
        Iterate through one epoch of data.
        
        Yields batches one at a time, loading shards as needed.
        Memory-efficient for large datasets.
        """
        # Shuffle shard order if enabled
        shard_order = list(range(self.num_shards))
        if self.shuffle:
            self._rng.shuffle(shard_order)
        
        for shard_idx in shard_order:
            shard_path = self.shard_files[shard_idx]
            shard_data = self._load_shard(shard_path)
            
            batches = self._create_batches_from_shard(shard_data)
            
            # Shuffle batches within shard if enabled
            if self.shuffle:
                batch_indices = list(range(len(batches)))
                self._rng.shuffle(batch_indices)
                batches = [batches[i] for i in batch_indices]
            
            for batch in batches:
                yield batch
            
            # Free memory after processing shard
            del shard_data
    
    def get_sample_batch(self) -> Dict[str, jnp.ndarray]:
        """Get a single sample batch for model initialization."""
        shard_data = self._load_shard(self.shard_files[0])
        batches = self._create_batches_from_shard(shard_data)
        return batches[0]
    
    def get_validation_batches(self, max_batches: int = 50) -> List[Dict[str, jnp.ndarray]]:
        """
        Load validation batches (limited number for memory efficiency).
        
        Args:
            max_batches: Maximum number of batches to load
            
        Returns:
            List of validation batches
        """
        val_batches = []
        batch_count = 0
        
        for batch in self.iter_epoch():
            val_batches.append(batch)
            batch_count += 1
            if batch_count >= max_batches:
                break
        
        return val_batches
    
    @property
    def num_batches_per_epoch(self) -> int:
        """Estimate number of batches per epoch."""
        return (self.total_samples + self.batch_size - 1) // self.batch_size


def create_synthetic_batches(
    num_batches: int, 
    batch_size: int, 
    seq_length: int, 
    vocab_size: int
) -> List[Dict[str, jnp.ndarray]]:
    """Create synthetic tensor batches for testing."""
    rng = jax.random.PRNGKey(42)
    batches = []
    
    for _ in range(num_batches):
        rng, key = jax.random.split(rng)
        input_ids = jax.random.randint(key, (batch_size, seq_length), 0, vocab_size)
        batches.append({
            "input_ids": input_ids,
            "targets": input_ids,
            "text": input_ids,
        })
    
    return batches


def main():
    """Main training function with CLI support."""
    args = parse_args()
    
    # Create configuration
    config = AGIConfig(
        # Core architecture
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=8000,
        
        # Advanced features
        multimodal_enabled=True,
        quantum_layers=2,
        max_reasoning_steps=8,
        consciousness_simulation=True,
        
        # Training settings
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        eval_interval=2,
    )
    
    # Load training data
    logger.info("Loading training data...")
    
    if args.data_dir and os.path.isdir(args.data_dir):
        # Use streaming data loader for memory efficiency
        try:
            train_loader = ShardedDataLoader(
                data_dir=args.data_dir,
                batch_size=config.batch_size,
                seq_length=config.max_seq_length,
                shuffle=True
            )
            
            # Create separate validation loader (no shuffle)
            val_loader = ShardedDataLoader(
                data_dir=args.data_dir,
                batch_size=config.batch_size,
                seq_length=config.max_seq_length,
                shuffle=False
            )
            
            logger.info(f"Streaming data loader ready: ~{train_loader.num_batches_per_epoch} batches/epoch")
            
            train_data = train_loader
            val_data = val_loader
            
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
    else:
        logger.warning("No data directory provided. Creating synthetic data for testing...")
        train_data = create_synthetic_batches(
            num_batches=100,
            batch_size=config.batch_size,
            seq_length=config.max_seq_length,
            vocab_size=config.vocab_size
        )
        val_data = create_synthetic_batches(
            num_batches=10,
            batch_size=config.batch_size,
            seq_length=config.max_seq_length,
            vocab_size=config.vocab_size
        )
    
    # Create trainer
    trainer = AGITrainer(config)
    
    # Handle checkpoint resumption
    if args.resume:
        if not os.path.exists(args.resume):
            logger.error(f"Checkpoint not found: {args.resume}")
            sys.exit(1)
        logger.info(f"Will resume training from: {args.resume}")
    
    # Start training
    results = trainer.train(
        train_data, 
        val_data,
        resume_checkpoint=args.resume
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Final training loss: {results['training_losses'][-1]:.4f}")
    
    if results['validation_losses']:
        logger.info(f"Final validation loss: {results['validation_losses'][-1]:.4f}")


if __name__ == "__main__":
    main()

