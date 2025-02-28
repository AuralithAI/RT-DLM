import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import gc
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_tms import TMSModel
from data_utils import DataProcessor, load_data, create_batches
from memory_bank import MemoryBank, ShortTermMemory, MidTermMemory
from train_config import TrainConfig

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('train.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

def train_and_evaluate(config, losses, similarity_scores, thought_logs):
    logger.info("Starting train_and_evaluate")
    rng = jax.random.PRNGKey(42)
    devices = jax.devices()
    logger.info(f"Detected devices: {devices}")

    ltm = MemoryBank(memory_size=config.memory_size, embedding_dim=config.d_model, retrieval_k=config.retrieval_k)
    stm = ShortTermMemory(buffer_size=config.stm_buffer_size, embedding_dim=config.d_model)
    mtm = MidTermMemory(buffer_size=config.mtm_buffer_size, embedding_dim=config.d_model, retention_steps=config.retention_steps)
    logger.info("Memory banks initialized")

    # Load dataset directly to jnp
    dataset_path = "data/train_data.txt"
    logger.info(f"Loading dataset from {dataset_path}")
    processor = DataProcessor(vocab_size=config.vocab_size)
    raw_texts = load_data(dataset_path)
    logger.info(f"Tokenizing {len(raw_texts)} texts")
    tokenized_texts = [processor.tokenize(text) for text in raw_texts]
    inputs = jnp.array([processor.pad_sequence(tokens, config.max_seq_length) for tokens in tokenized_texts], dtype=jnp.int32)
    targets = inputs  # Next-token prediction
    logger.info(f"Dataset loaded - {inputs.shape[0]} samples")

    # Model definition with dynamic pruning
    def forward_fn(inputs, return_attention=False, retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None, **kwargs):
        model = TMSModel(
            d_model=config.d_model, 
            num_heads=config.num_heads, 
            num_layers=config.num_layers,
            vocab_size=config.vocab_size, 
            max_seq_length=config.max_seq_length,
            moe_experts=config.moe_experts, 
            moe_top_k=config.moe_top_k,
            memory_size=config.memory_size, 
            retrieval_k=config.retrieval_k,
            ltm_weight=config.ltm_weight, 
            stm_weight=config.stm_weight, 
            mtm_weight=config.mtm_weight
        )
        return model(inputs, return_attention=return_attention, retrieved_memory_ltm=retrieved_memory_ltm, 
                     retrieved_memory_stm=retrieved_memory_stm, retrieved_memory_mtm=retrieved_memory_mtm, **kwargs)

    logger.info("Transforming model with Haiku")
    model = hk.transform_with_state(forward_fn)
    init_batch_size = min(8, config.batch_size)
    logger.info(f"Initializing model with batch size {init_batch_size}")
    inputs_init = inputs[:init_batch_size]
    params, state = model.init(rng, inputs_init)
    logger.info("Model initialized")

    # Meta-learning optimizers
    logger.info("Setting up optimizers")
    meta_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,                   
        peak_value=config.learning_rate,  
        warmup_steps=config.warmup_steps, 
        decay_steps=config.decay_steps,   
        end_value=config.learning_rate * 0.1  
    )
    meta_optimizer = optax.adam(learning_rate=meta_schedule)
    
    inner_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,                        
        peak_value=config.inner_learning_rate, 
        warmup_steps=config.num_inner_steps * 10, 
        decay_steps=config.num_inner_steps * 100, 
        end_value=config.inner_learning_rate * 0.1  
    )
    inner_optimizer = optax.sgd(learning_rate=inner_schedule)
    
    meta_opt_state = meta_optimizer.init(params)
    inner_opt_state = inner_optimizer.init(params)
    logger.info("Optimizers initialized with warm-up cosine decay schedules")

    # Loss functions
    def compute_loss(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm):
        inputs = jnp.asarray(inputs, dtype=jnp.int32, copy=True)
        targets = jnp.asarray(targets, dtype=jnp.int32, copy=True)
        (logits, (attn_weights_self, attn_weights_transformer), expert_indices, aux_loss), new_state = model.apply(
            params, state, rng, inputs, return_attention=True,
            retrieved_memory_ltm=retrieved_memory_ltm, retrieved_memory_stm=retrieved_memory_stm, retrieved_memory_mtm=retrieved_memory_mtm,
            spike_threshold=config.spike_threshold, epsilon=config.EPSILON
        )
        embeddings = get_embeddings(config, params, state, rng, inputs, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm, config.spike_threshold, config.EPSILON)
        query_key = jnp.mean(embeddings, axis=1)
        ltm_norm = jnp.linalg.norm(retrieved_memory_ltm, axis=-1, keepdims=True) + config.EPSILON
        stm_norm = jnp.linalg.norm(retrieved_memory_stm, axis=-1, keepdims=True) + config.EPSILON
        mtm_norm = jnp.linalg.norm(retrieved_memory_mtm, axis=-1, keepdims=True) + config.EPSILON
        query_norm = jnp.linalg.norm(query_key, axis=-1, keepdims=True) + config.EPSILON
        ltm_similarity = jnp.sum(query_key * retrieved_memory_ltm, axis=-1) / (query_norm * ltm_norm)
        stm_similarity = jnp.sum(query_key * retrieved_memory_stm, axis=-1) / (query_norm * stm_norm)
        mtm_similarity = jnp.sum(query_key * retrieved_memory_mtm, axis=-1) / (query_norm * mtm_norm)
        similarity_score = jnp.stack([ltm_similarity, stm_similarity, mtm_similarity], axis=-1)
        similarity_score = jax.lax.stop_gradient(similarity_score)
        similarity_score = jnp.nan_to_num(similarity_score, nan=0.0, posinf=1.0, neginf=-1.0)
        similarity_score = jnp.clip(similarity_score, -1.0, 1.0)

        thoughts = {
            "attn_weights_self": attn_weights_self,
            "attn_weights_transformer": attn_weights_transformer,
            "similarity_score": similarity_score,
            "expert_indices": expert_indices
        }

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        logger.info(f"Cross-entropy loss: {float(loss):.4f}, Aux loss: {float(aux_loss):.4f}, Total loss: {float(loss + 0.001 * aux_loss):.4f}")
        logger.info(f"Logits norm: {jnp.linalg.norm(logits):.4f}, Targets norm: {jnp.linalg.norm(targets):.4f}")
        logger.info(f"Similarity score: {jnp.mean(similarity_score):.4f}")
        total_loss = loss + 0.001 * aux_loss
        return total_loss, new_state, thoughts

    # Loss for gradients
    def loss_for_gradients(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm):
        loss, _, _ = compute_loss(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm)
        return loss

    # Inner loop train step with gradient accumulation and pruning
    @jax.jit
    def _accumulate_gradients(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm):
        loss, new_state, thoughts = compute_loss(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm)
        grads = jax.grad(loss_for_gradients)(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm)
        return grads, loss, new_state, thoughts

    def train_step(params, state, inner_opt_state, rng, inputs, targets, step, accum_steps=2, prune_interval=50):
        embeddings = jnp.mean(get_embeddings(config, params, state, rng, inputs, spike_threshold=config.spike_threshold, epsilon=config.EPSILON), axis=1)
        query_key_np = np.asarray(jax.device_get(embeddings), dtype=np.float32)
        ltm_memory = jnp.array(ltm.retrieve(query_key_np, config.spike_threshold, config.EPSILON), dtype=config.embedding_dtype)
        stm_memory = jnp.array(stm.retrieve(query_key_np, config.spike_threshold, config.EPSILON), dtype=config.embedding_dtype)
        mtm_memory = jnp.array(mtm.retrieve(query_key_np, config.spike_threshold, config.EPSILON), dtype=config.embedding_dtype)
        logger.info(f"LTM norm: {jnp.linalg.norm(ltm_memory):.4f}, STM norm: {jnp.linalg.norm(stm_memory):.4f}, MTM norm: {jnp.linalg.norm(mtm_memory):.4f}")

        batch_size = inputs.shape[0]
        mini_batch_size = batch_size // accum_steps
        total_grads = jax.tree.map(lambda x: jnp.zeros_like(x), params)
        total_loss = 0.0
        new_state = state
        thoughts = None

        for i in range(accum_steps):
            start_idx = i * mini_batch_size
            end_idx = (i + 1) * mini_batch_size if i < accum_steps - 1 else batch_size
            mini_inputs = inputs[start_idx:end_idx]
            mini_targets = targets[start_idx:end_idx]

            grads, loss, new_state, mini_thoughts = _accumulate_gradients(
                params, new_state, rng, mini_inputs, mini_targets, ltm_memory[start_idx:end_idx],
                stm_memory[start_idx:end_idx], mtm_memory[start_idx:end_idx]
            )
            total_grads = jax.tree.map(lambda x, y: x + y, total_grads, grads)
            total_loss += loss * (mini_inputs.shape[0] / batch_size)
            thoughts = mini_thoughts if thoughts is None else thoughts  # Keep last thoughts

        total_grads = jax.tree.map(lambda x: x / accum_steps, total_grads)
        # Gradient clipping
        total_grads = jax.tree.map(lambda x: jnp.clip_by_norm(x, 10.0), total_grads)
        updates, new_inner_opt_state = inner_optimizer.update(total_grads, inner_opt_state, params)
        params = optax.apply_updates(params, updates)

        # Prune periodically (every prune_interval steps)
        if step % prune_interval == 0 and step > 0:
            logger.info(f"Pruning at step {step}...")
            # Prune attention heads and experts
            new_params, new_state = _prune_model(params, state, rng, inputs[:1])  # Use a small batch for pruning
            params = new_params
            state = new_state

        return total_loss, new_state, thoughts, new_inner_opt_state

    def _prune_model(params, state, rng, inputs):
        """
        Prune underutilized attention heads, experts, and Transformer components from the model, with proper Haiku state management.
        """
        # Define forward pass for pruning (to update usage stats)
        def forward_prune(inputs, return_attention=False, **kwargs):
            model = TMSModel(
                d_model=config.d_model, 
                num_heads=config.num_heads, 
                num_layers=config.num_layers,
                vocab_size=config.vocab_size, 
                max_seq_length=config.max_seq_length,
                moe_experts=config.moe_experts, 
                moe_top_k=config.moe_top_k,
                memory_size=config.memory_size, 
                retrieval_k=config.retrieval_k,
                ltm_weight=config.ltm_weight, 
                stm_weight=config.stm_weight, 
                mtm_weight=config.mtm_weight
            )
            return model(inputs, return_attention=return_attention, **kwargs)

        # Transform with state for pruning
        prune_model = hk.transform_with_state(forward_prune)
        _, new_state = prune_model.apply(params, state, rng, inputs, return_attention=True,
                                        retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None,
                                        spike_threshold=config.spike_threshold, epsilon=config.EPSILON)

        # Prune SelfAttentionModel
        self_attention = new_state["self_attention"]
        if hasattr(self_attention, "prune_heads"):
            new_self_attention = self_attention.prune_heads(threshold=config.prune_threshold)
            new_self_attention_params, new_self_attention_state = hk.transform_with_state(
                lambda x: new_self_attention(x, return_attention=True)
            ).init(rng, inputs[:1])
            params["self_attention"] = new_self_attention_params
            state["self_attention"] = new_self_attention_state

        # Prune SparseMoE
        moe = new_state["moe"]
        if hasattr(moe, "prune_experts"):
            new_moe = moe.prune_experts(threshold=config.prune_threshold)
            new_moe_params, new_moe_state = hk.transform_with_state(
                lambda x: new_moe(x)
            ).init(rng, inputs[:1])
            params["moe"] = new_moe_params
            state["moe"] = new_moe_state

        # Prune TransformerModel
        transformer = new_state["transformer"]
        if hasattr(transformer, "prune_layers"):
            new_transformer = transformer.prune_layers(threshold=config.prune_threshold)
            new_transformer_params, new_transformer_state = hk.transform_with_state(
                lambda x: new_transformer(x, return_attention=True)
            ).init(rng, inputs[:1])
            params["transformer"] = new_transformer_params
            state["transformer"] = new_transformer_state

        return params, state

    # Meta-step: Outer loop (MAML update)
    def meta_step(params, state, meta_opt_state, inner_opt_state, rng, support_inputs, support_targets, query_inputs, query_targets, accum_steps=2):
        adapted_params = params
        adapted_state = state
        new_inner_opt_state = inner_opt_state
        num_inner_steps = config.num_inner_steps 

        for _ in range(num_inner_steps):
            loss, adapted_state, support_thoughts, new_inner_opt_state = train_step(
                adapted_params, adapted_state, new_inner_opt_state, rng, support_inputs, support_targets, 0, accum_steps=accum_steps, prune_interval=50
            )
            adapted_params = jax.tree.map(lambda p: p, adapted_params)

        query_embeddings = jnp.mean(get_embeddings(config, params, adapted_state, rng, query_inputs, spike_threshold=config.spike_threshold, epsilon=config.EPSILON), axis=1)
        query_key_np = np.asarray(jax.device_get(query_embeddings), dtype=config.embedding_dtype)
        ltm_memory_query = jnp.array(ltm.retrieve(query_key_np, config.spike_threshold, config.EPSILON), dtype=config.embedding_dtype)
        stm_memory_query = jnp.array(stm.retrieve(query_key_np, config.spike_threshold, config.EPSILON), dtype=config.embedding_dtype)
        mtm_memory_query = jnp.array(mtm.retrieve(query_key_np, config.spike_threshold, config.EPSILON), dtype=config.embedding_dtype)
        
        loss, new_state, query_thoughts = compute_loss(
            adapted_params, adapted_state, rng, query_inputs, query_targets, ltm_memory_query, stm_memory_query, mtm_memory_query
        )

        logger.info(f"Aux loss: {query_thoughts['expert_indices'].mean():.4f}, Total loss: {loss:.4f}")
        
        grads = jax.grad(loss_for_gradients)(params, state, rng, query_inputs, query_targets, ltm_memory_query, stm_memory_query, mtm_memory_query)
        updates, new_meta_opt_state = meta_optimizer.update(grads, meta_opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return loss, params, new_state, new_meta_opt_state, support_thoughts, query_thoughts, new_inner_opt_state

    task_size = config.task_size
    num_tasks = inputs.shape[0] // task_size
    logger.info(f"Creating task batches with task_size={task_size} (Total tasks: {num_tasks})")
    task_batches = create_batches(inputs, targets, task_size, shuffle=True)

    for task_idx, (task_inputs, task_targets) in enumerate(task_batches):
        support_inputs = task_inputs[:5]
        support_targets = task_targets[:5]
        query_inputs = task_inputs[5:]
        query_targets = task_targets[5:]

        if support_inputs.shape[0] < 5 or query_inputs.shape[0] < 1:
            logger.info(f"Skipping task {task_idx+1} - insufficient support/query samples")
            continue

        logger.info(f"Processing task {task_idx+1}/{num_tasks} ({(task_idx + 1) / num_tasks * 100:.1f}% complete)")
        step_rng, rng = jax.random.split(rng)
        try:
            loss, params, state, meta_opt_state, support_thoughts, query_thoughts, inner_opt_state = meta_step(
                params, state, meta_opt_state, inner_opt_state, step_rng, support_inputs, support_targets, query_inputs, query_targets, accum_steps=2
            )
        except Exception as e:
            logger.error(f"[Task {task_idx+1}] Failed: {e}")
            continue

        losses.append(float(loss))

        embeddings = jnp.mean(get_embeddings(config, params, state, step_rng, query_inputs, spike_threshold=config.spike_threshold, epsilon=config.EPSILON), axis=1)
        ltm.store(embeddings, embeddings, config.spike_threshold, config.EPSILON)
        stm.store(embeddings, embeddings, config.spike_threshold, config.EPSILON)
        mtm.store(embeddings, embeddings, config.spike_threshold, config.EPSILON)

        thought_logs.append({"task_idx": task_idx, "support_thoughts": support_thoughts, "query_thoughts": query_thoughts})
        similarity_score = thought_logs[-1]["query_thoughts"]["similarity_score"]
        similarity_score_mean = float(jnp.mean(similarity_score))
        similarity_scores.append(similarity_score_mean)

        logger.info(f"[Task {task_idx+1}] completed - Loss: {loss:.4f} | Similarity: {similarity_score_mean:.4f} | LR: {config.learning_rate:.6f} | Inner LR: {config.inner_learning_rate:.6f}")
        if task_idx % 100 == 0:
            clear_gpu_memory()
        gc.collect()
        jax.clear_caches()

    logger.info("Training completed")
    return losses, params, similarity_scores, state, ltm, stm, mtm, thought_logs

def clear_gpu_memory():
    """Clear GPU memory after each trial."""
    jax.clear_caches()  
    for device in jax.devices():
        try:
            jax.device_put(jnp.zeros((1,)), device=device) 
            jax.device_put(None, device=device) 
        except Exception as e:
            logger.warning(f"[WARNING] Failed to clear device memory: {e}")
    gc.collect()
    logger.info("[INFO] GPU memory cleared")

def get_embeddings(config, params, state, rng, inputs, retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None, spike_threshold=0.1, epsilon=1e-8):
    def forward_with_embeddings(inputs, return_attention=False, retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None, spike_threshold=spike_threshold, epsilon=epsilon):
        if not 0 <= spike_threshold <= 1:
            raise ValueError("spike_threshold must be between 0 and 1")
        inputs = jnp.asarray(inputs, dtype=jnp.int32, copy=True)
        model = TMSModel(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length,
            moe_experts=config.moe_experts,
            moe_top_k=config.moe_top_k,
            memory_size=config.memory_size,
            retrieval_k=config.retrieval_k,
            ltm_weight=config.ltm_weight,
            stm_weight=config.stm_weight,
            mtm_weight=config.mtm_weight
        )
        x = model.embedding(inputs) + model.position_enc(jnp.arange(inputs.shape[1]))
        if retrieved_memory_ltm is not None:
            retrieved_memory_ltm = model.memory_projection_ltm(jnp.repeat(jnp.expand_dims(retrieved_memory_ltm, axis=1), x.shape[1], axis=1))
            x += model.ltm_weight * retrieved_memory_ltm
        if retrieved_memory_stm is not None:
            retrieved_memory_stm = model.memory_projection_stm(jnp.repeat(jnp.expand_dims(retrieved_memory_stm, axis=1), x.shape[1], axis=1))
            x += model.stm_weight * retrieved_memory_stm
        if retrieved_memory_mtm is not None:
            retrieved_memory_mtm = model.memory_projection_mtm(jnp.repeat(jnp.expand_dims(retrieved_memory_mtm, axis=1), x.shape[1], axis=1))
            x += model.mtm_weight * retrieved_memory_mtm
        x, _ = model.self_attention(inputs, return_attention=True, spike_threshold=spike_threshold, epsilon=epsilon)
        x, _ = model.transformer(x, rng, return_attention=True, spike_threshold=spike_threshold, epsilon=epsilon)
        x, _, _ = model.moe(x, spike_threshold=spike_threshold, epsilon=epsilon)
        x = model.norm(x)
        return x

    transformed = hk.transform_with_state(forward_with_embeddings)
    embeddings, _ = transformed.apply(params, state, rng, inputs, retrieved_memory_ltm=retrieved_memory_ltm, 
                                     retrieved_memory_stm=retrieved_memory_stm, retrieved_memory_mtm=retrieved_memory_mtm)
    return embeddings

if __name__ == "__main__":
    config = TrainConfig(
        d_model=768,
        num_heads=12,
        num_layers=10,
        moe_experts=4,
        moe_top_k=2,
        batch_size=4,
        learning_rate=0.0005,
        inner_learning_rate=0.0001,
        warmup_steps=3000,
        decay_steps=300000,
        memory_size=10000,
        retrieval_k=5,
        stm_buffer_size=8,
        mtm_buffer_size=500,
        retention_steps=100,
        ltm_weight=0.4,
        stm_weight=0.3,
        mtm_weight=0.3,
        spike_threshold=0.3,
        epsilon=1e-6,
        prune_threshold=0.01
    )
    losses, params, similarity_scores, state, ltm, stm, mtm, thought_logs = train_and_evaluate(config, [], [], [])
    logger.info(f"Training completed - Final Loss: {losses[-1]:.4f}, Final Similarity: {similarity_scores[-1]:.4f}")