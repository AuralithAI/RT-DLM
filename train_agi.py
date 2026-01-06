import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
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

from rtdlm_agi_complete import (
    create_rtdlm_agi, create_agi_optimizer, 
    compute_agi_loss
)
from config.agi_config import AGIConfig
from data_processing.data_utils import DataProcessor, load_data, create_batches
from advanced_learning.advanced_algorithms import (
    ContinualLearner, TaskMemory, compute_ewc_loss, compute_fisher_information
)

def cosine_similarity(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compute cosine similarity between two vectors"""
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = jnp.linalg.norm(a_flat) + 1e-8
    norm_b = jnp.linalg.norm(b_flat) + 1e-8
    return jnp.sum(a_flat * b_flat) / (norm_a * norm_b)

class AGITrainer:
    """Advanced trainer for RT-DLM AGI with comprehensive capabilities"""
    
    def __init__(self, config: AGIConfig):
        self.config = config
        self.rng = jax.random.PRNGKey(42)
        
        # Initialize model
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
        self.lambda_ewc = 1000.0  # EWC regularization strength
        self.enable_continual_learning = True
        
        # Data processor
        self.data_processor = DataProcessor(
            vocab_size=config.vocab_size,
            model_prefix="data/rt_dlm_sp"
        )
        
    def initialize_model(self, sample_batch):
        """Initialize model parameters"""
        print("[INFO] Initializing RT-DLM AGI model...")
        
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
        print(f"[INFO] Model initialized with {param_count:,} parameters")
        
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
        if not self.enable_continual_learning or self.params is None:
            return
            
        print(f"[INFO] Consolidating task {task_id} with EWC...")
        
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
            importance_weights=fisher_matrix,  # Use Fisher as importance for SI
            performance_metrics={"final_loss": float(self.training_losses[-1]) if self.training_losses else 0.0},
            num_samples=len(data_samples)
        )
        
        self.task_memories.append(task_memory)
        print(f"[INFO] Task {task_id} consolidated. Total task memories: {len(self.task_memories)}")
    
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

    def create_training_batch(self, texts: List[str], include_multimodal: bool = False,
                               images: Optional[np.ndarray] = None,
                               audio: Optional[np.ndarray] = None):
        """Create a training batch with optional multimodal data
        
        Args:
            texts: List of text strings to tokenize
            include_multimodal: Whether to include synthetic multimodal data
            images: Optional real image data (batch, H, W, C) in uint8 [0-255]
            audio: Optional real audio mel spectrograms (batch, time, freq)
        
        Returns:
            Dictionary with input_ids, targets, and optional multimodal_inputs
        """
        # Tokenize texts
        tokenized = [self.data_processor.tokenize(text) for text in texts]
        
        # Pad sequences
        max_len = min(self.config.max_seq_length, max(len(tokens) for tokens in tokenized))
        inputs = jnp.array([
            self.data_processor.pad_sequence(tokens, max_len) 
            for tokens in tokenized
        ], dtype=jnp.int32)
        
        batch = {
            "input_ids": inputs,
            "targets": inputs,  # For next-token prediction
            "text": inputs,  # Alias for multimodal batch format
        }
        
        batch_size = inputs.shape[0]
        
        # Add multimodal data if provided or generate synthetic
        if include_multimodal and self.config.multimodal_enabled:
            self.rng, *modal_rngs = jax.random.split(self.rng, 4)
            
            # Use provided images or generate synthetic
            if images is not None:
                # Normalize real images from [0-255] to [0-1]
                batch_images = jnp.array(images[:batch_size] / 255.0, dtype=jnp.float32)
                # Resize if needed
                if batch_images.shape[1:3] != (224, 224):
                    batch_images = jax.image.resize(
                        batch_images, 
                        (batch_size, 224, 224, 3),
                        method='bilinear'
                    )
            else:
                # Generate synthetic image data
                batch_images = jax.random.normal(modal_rngs[0], (batch_size, 224, 224, 3)) * 0.1
            
            # Use provided audio or generate synthetic mel spectrograms
            if audio is not None:
                mel_spectrograms = jnp.array(audio[:batch_size], dtype=jnp.float32)
                # Resize if needed
                if mel_spectrograms.shape[1:] != (128, 128):
                    mel_spectrograms = jax.image.resize(
                        mel_spectrograms[..., None],  # Add channel dim
                        (batch_size, 128, 128, 1),
                        method='bilinear'
                    )[..., 0]  # Remove channel dim
            else:
                # Generate synthetic mel spectrogram
                mel_spectrograms = jax.random.normal(modal_rngs[1], (batch_size, 128, 128)) * 0.1
            
            batch["image"] = batch_images
            batch["audio"] = mel_spectrograms
            
            batch["multimodal_inputs"] = {
                "images": batch_images,
                "audio": mel_spectrograms,
                "video": jax.random.normal(modal_rngs[2], (batch_size, 8, 112, 112, 3)) * 0.1,
            }
        
        return batch
    
    def create_reasoning_task(self, batch_size: int = 8):
        """Create a reasoning task for training reasoning capabilities"""
        
        # Simple arithmetic reasoning task
        reasoning_problems = []
        for _ in range(batch_size):
            a = np.random.randint(1, 100)
            b = np.random.randint(1, 100)
            operation = np.random.choice(['+', '-', '*'])
            
            if operation == '+':
                answer = a + b
                problem = f"What is {a} plus {b}? Let me think step by step."
            elif operation == '-':
                answer = a - b  
                problem = f"What is {a} minus {b}? Let me think step by step."
            else:  # multiplication
                answer = a * b
                problem = f"What is {a} times {b}? Let me think step by step."
            
            reasoning_problems.append(problem)
        
        return self.create_training_batch(reasoning_problems, include_multimodal=False)
    
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
        (total_loss, (model_output, base_loss, ewc_loss)), grads = jax.value_and_grad(
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

    def evaluate_model(self, eval_data: List[str], num_samples: int = 100):
        """Comprehensive model evaluation"""
        print("[INFO] Evaluating model...")
        
        eval_losses = []
        reasoning_scores = []
        consciousness_scores = []
        
        # Regular evaluation
        for i in range(0, min(len(eval_data), num_samples), self.config.batch_size):
            batch_texts = eval_data[i:i + self.config.batch_size]
            if len(batch_texts) < self.config.batch_size:
                break
                
            batch = self.create_training_batch(batch_texts)
            
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
        reasoning_batch = self.create_reasoning_task(self.config.batch_size)
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
            step_pattern = r'[Ss]tep\s*\d+:\s*(.+?)(?=[Ss]tep\s*\d+:|[Aa]nswer:|$)'
            answer_pattern = r'[Aa]nswer:\s*(.+?)(?:\.|$)'
            
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
        """Save training checkpoint"""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            "params": self.params,
            "opt_state": self.opt_state,
            "epoch": epoch,
            "step_count": self.step_count,
            "config": self.config.to_dict(),
            "metrics": metrics,
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f"rtdlm_agi_epoch_{epoch}.pkl")
        
        # Save using JAX's serialization
        import pickle
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"[INFO] Checkpoint saved: {checkpoint_path}")
    
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
    
    def train(self, train_data: List[str], val_data: List[str]):
        """Complete training loop for RT-DLM AGI"""
        print("=" * 60)
        print("RT-DLM AGI Training Started")
        print("=" * 60)
        
        # Print configuration
        self.config.print_summary()
        
        # Initialize model with first batch
        sample_batch = self.create_training_batch(train_data[:self.config.batch_size])
        self.initialize_model(sample_batch)
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 5
        
        for epoch in range(self.config.num_epochs):
            print(f"\n[EPOCH {epoch + 1}/{self.config.num_epochs}]")
            
            epoch_losses = []
            epoch_start_time = time.time()
            
            # Training
            num_batches = len(train_data) // self.config.batch_size
            for batch_idx in range(0, len(train_data), self.config.batch_size):
                if batch_idx + self.config.batch_size > len(train_data):
                    break
                
                # Create batch
                batch_texts = train_data[batch_idx:batch_idx + self.config.batch_size]
                batch = self.create_training_batch(
                    batch_texts, 
                    include_multimodal=self.config.multimodal_enabled
                )
                
                # Training step
                self.rng, train_rng = jax.random.split(self.rng)
                self.params, self.opt_state, loss, model_output = self.train_step(
                    self.params, self.opt_state, batch, train_rng
                )
                
                # NaN check for loss
                if jnp.isnan(loss) or jnp.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping...")
                    continue
                
                epoch_losses.append(float(loss))
                self.training_losses.append(float(loss))
                self.step_count += 1
                
                # Log progress
                if batch_idx // self.config.batch_size % 50 == 0:
                    current_batch = batch_idx // self.config.batch_size + 1
                    print(f"  Batch {current_batch}/{num_batches}, Loss: {loss:.4f}")
            
            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = np.mean(epoch_losses)
            
            print(f"  Average Loss: {avg_epoch_loss:.4f}")
            print(f"  Epoch Time: {epoch_time:.1f}s")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0 and epoch > 0:
                self.save_checkpoint(epoch, {"avg_loss": avg_epoch_loss})
                logger.info(f"Periodic checkpoint saved at epoch {epoch}")
            
            # Validation
            if epoch % self.config.eval_interval == 0 or epoch == self.config.num_epochs - 1:
                metrics = self.evaluate_model(val_data)
                
                self.validation_losses.append(metrics["eval_loss"])
                self.reasoning_accuracies.append(metrics["reasoning_accuracy"])
                self.consciousness_coherence.append(metrics["consciousness_coherence"])
                
                print(f"  Validation Loss: {metrics['eval_loss']:.4f}")
                print(f"  Reasoning Accuracy: {metrics['reasoning_accuracy']:.4f}")
                print(f"  Consciousness Coherence: {metrics['consciousness_coherence']:.4f}")
                
                # Early stopping check
                if metrics["eval_loss"] < best_val_loss:
                    best_val_loss = metrics["eval_loss"]
                    patience_counter = 0
                    self.save_checkpoint(epoch, metrics)
                else:
                    patience_counter += 1
                
                if patience_counter >= max_patience:
                    print(f"[INFO] Early stopping triggered after {patience_counter} epochs without improvement")
                    break
        
        print("\n" + "=" * 60)
        print("RT-DLM AGI Training Completed!")
        print("=" * 60)
        
        # Plot training metrics
        self.plot_training_metrics()
        
        return {
            "final_params": self.params,
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "reasoning_accuracies": self.reasoning_accuracies,
            "consciousness_coherence": self.consciousness_coherence,
        }

def main():
    """Main training function"""
    # Create AGI configuration
    config = AGIConfig(
        # Core architecture
        d_model=512,
        num_heads=8,
        num_layers=12,
        vocab_size=8000,
        
        # Advanced features
        multimodal_enabled=True,
        quantum_layers=2,
        max_reasoning_steps=8,
        consciousness_simulation=True,
        
        # Training settings
        batch_size=16,
        num_epochs=10,
        learning_rate=3e-4,
        eval_interval=2,
    )
    
    # Load training data
    print("[INFO] Loading training data...")
    try:
        train_data = load_data("data/train_data.txt")
        val_data = load_data("data/validation_data.txt")
        
        # Sample smaller datasets for faster training
        train_data = train_data[:10000]  # Use first 10k samples
        val_data = val_data[:1000]      # Use first 1k validation samples
        
        print(f"[INFO] Loaded {len(train_data)} training samples, {len(val_data)} validation samples")
        
    except FileNotFoundError:
        print("[WARNING] Training data not found. Creating synthetic data...")
        # Create synthetic training data
        train_data = [
            f"This is sample text number {i} for training the AGI model."
            for i in range(1000)
        ]
        val_data = [
            f"This is validation text number {i} for evaluating the AGI model."
            for i in range(100)
        ]
    
    # Create trainer
    trainer = AGITrainer(config)
    
    # Start training
    results = trainer.train(train_data, val_data)
    
    print(f"[INFO] Training completed successfully!")
    print(f"[INFO] Final training loss: {results['training_losses'][-1]:.4f}")
    
    if results['validation_losses']:
        print(f"[INFO] Final validation loss: {results['validation_losses'][-1]:.4f}")

if __name__ == "__main__":
    main()
