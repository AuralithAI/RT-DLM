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
from data_utils import load_multimodal_data, create_multimodal_batches
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

    multimodal_datasets = load_multimodal_data("data/", config)
    logger.info(f"Loaded {len(multimodal_datasets)} modality datasets")

    def forward_fn(inputs, modality_types, output_modality, return_attention=False, 
                   retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None, **kwargs):
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
            mtm_weight=config.mtm_weight,
            audio_sample_rate=config.audio_sample_rate,
            image_size=config.image_size
        )
        return model(inputs, modality_types, output_modality, return_attention=return_attention,
                     retrieved_memory_ltm=retrieved_memory_ltm, retrieved_memory_stm=retrieved_memory_stm,
                     retrieved_memory_mtm=retrieved_memory_mtm, **kwargs)

    logger.info("Transforming model with Haiku")
    model = hk.transform_with_state(forward_fn)

    # Initialize with a small batch
    init_inputs = [jnp.ones((min(8, config.batch_size), config.max_seq_length), dtype=jnp.int32)]
    init_modality_types = ("text",)
    init_output_modality = "text"
    params, state = model.init(rng, init_inputs, init_modality_types, init_output_modality)
    logger.info("Model initialized")

    # Optimizers
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
    logger.info("Optimizers initialized")

    def compute_loss(params, state, rng, inputs, modality_types, targets, output_modality, 
                     retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm):
        (output, (attn_weights_self, attn_weights_transformer), expert_indices, aux_loss), new_state = model.apply(
            params, state, rng, inputs, modality_types, output_modality, return_attention=True,
            retrieved_memory_ltm=retrieved_memory_ltm, retrieved_memory_stm=retrieved_memory_stm,
            retrieved_memory_mtm=retrieved_memory_mtm, spike_threshold=config.spike_threshold, epsilon=config.EPSILON
        )
        embeddings = get_embeddings(config, params, state, rng, inputs, modality_types, retrieved_memory_ltm, 
                                   retrieved_memory_stm, retrieved_memory_mtm, config.spike_threshold, config.EPSILON)
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

        # Modality-specific loss
        if output_modality == "text":
            loss = optax.softmax_cross_entropy_with_integer_labels(output, targets).mean()
            logger.info(f"Text Loss: {float(loss):.4f}")
        elif output_modality in ["audio", "image", "video"]:
            loss = jnp.mean((output - targets) ** 2)
            logger.info(f"{output_modality.capitalize()} Loss: {float(loss):.4f}")
        else:
            raise ValueError(f"Unsupported output modality: {output_modality}")

        total_loss = loss + 0.001 * aux_loss
        return total_loss, new_state, thoughts, {
            "loss": loss,
            "aux_loss": aux_loss,
            "output_norm": jnp.linalg.norm(output),
            "targets_norm": jnp.linalg.norm(targets),
            "similarity_mean": jnp.mean(similarity_score)
        }

    def loss_for_gradients(params, state, rng, inputs, modality_types, targets, output_modality, 
                           retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm):
        loss, _, _, _ = compute_loss(params, state, rng, inputs, modality_types, targets, output_modality,
                                     retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm)
        return loss

    def _accumulate_gradients(params, state, rng, inputs, modality_types, targets, output_modality,
                              retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm):
        modality_types = tuple(modality_types)
        loss, new_state, thoughts, metrics = compute_loss(params, state, rng, inputs, modality_types, targets, output_modality,
                                                         retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm)
        grads = jax.grad(loss_for_gradients)(params, state, rng, inputs, modality_types, targets, output_modality,
                                             retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm)
        return grads, loss, new_state, thoughts, metrics
    
    _accumulate_gradients = jax.jit(_accumulate_gradients, static_argnums=(4, 6))

    def train_step(params, state, inner_opt_state, rng, inputs, modality_types, targets, output_modality, step, accum_steps=2, prune_interval=50):
        modality_types = tuple(modality_types)
        embeddings = jnp.mean(get_embeddings(config, params, state, rng, inputs, modality_types, 
                                            spike_threshold=config.spike_threshold, epsilon=config.EPSILON), axis=1)
        query_key_np = np.asarray(jax.device_get(embeddings), dtype=np.float32)
        ltm_memory = jnp.array(ltm.retrieve(query_key_np, config.spike_threshold, config.EPSILON), dtype=config.embedding_dtype)
        stm_memory = jnp.array(stm.retrieve(query_key_np, config.spike_threshold, config.EPSILON), dtype=config.embedding_dtype)
        mtm_memory = jnp.array(mtm.retrieve(query_key_np, config.spike_threshold, config.EPSILON), dtype=config.embedding_dtype)

        batch_size = inputs[0].shape[0]  # Assume first modality defines batch size
        mini_batch_size = batch_size // accum_steps
        total_grads = jax.tree.map(lambda x: jnp.zeros_like(x), params)
        total_loss = 0.0
        new_state = state
        thoughts = None
        metrics = None

        for i in range(accum_steps):
            start_idx = i * mini_batch_size
            end_idx = (i + 1) * mini_batch_size if i < accum_steps - 1 else batch_size
            mini_inputs = [inp[start_idx:end_idx] for inp in inputs]
            mini_targets = targets[start_idx:end_idx] if targets is not None else None

            grads, loss, new_state, mini_thoughts, mini_metrics = _accumulate_gradients(
                params, new_state, rng, mini_inputs, modality_types, mini_targets, output_modality,
                ltm_memory[start_idx:end_idx], stm_memory[start_idx:end_idx], mtm_memory[start_idx:end_idx]
            )
            total_grads = jax.tree.map(lambda x, y: x + y, total_grads, grads)
            total_loss += loss * (mini_inputs[0].shape[0] / batch_size)
            thoughts = mini_thoughts if thoughts is None else thoughts
            metrics = mini_metrics if metrics is None else metrics

        logger.info(f"Loss: {float(jax.device_get(total_loss)):.4f}, Similarity: {float(jax.device_get(metrics['similarity_mean'])):.4f}")
        
        total_grads = jax.tree.map(lambda x: x / accum_steps, total_grads)
        total_grads = clip_gradients(config, total_grads, max_norm=10.0)
        updates, new_inner_opt_state = inner_optimizer.update(total_grads, inner_opt_state, params)
        params = optax.apply_updates(params, updates)

        if step % prune_interval == 0 and step > 0:
            logger.info(f"Pruning at step {step}...")
            new_params, new_state = _prune_model(params, state, rng, inputs[:1], modality_types)  # Small batch for pruning
            params = new_params
            state = new_state

        return total_loss, new_state, thoughts, new_inner_opt_state

    def _prune_model(params, state, rng, inputs, modality_types):
        modality_types = tuple(modality_types)
        def forward_prune(inputs, modality_types, output_modality, return_attention=False, **kwargs):
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
                mtm_weight=config.mtm_weight,
                audio_sample_rate=config.audio_sample_rate,
                image_size=config.image_size
            )
            return model(inputs, modality_types, output_modality, return_attention=return_attention, **kwargs)

        prune_model = hk.transform_with_state(forward_prune)
        _, new_state = prune_model.apply(params, state, rng, inputs, modality_types, "text", return_attention=True,
                                        retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None,
                                        spike_threshold=config.spike_threshold, epsilon=config.EPSILON)

        # Prune components
        self_attention = new_state["self_attention"]
        new_self_attention = self_attention.prune_heads_and_ffn(head_threshold=config.prune_threshold, ffn_threshold=config.prune_threshold)
        new_params, new_state_sa = hk.transform_with_state(
                                        lambda x: new_self_attention(x, return_attention=True)
                                    ).init(rng, inputs[:1])
        params["self_attention"] = new_params
        state["self_attention"] = new_state_sa

        moe = new_state["moe"]
        new_moe = moe.prune_experts_and_neurons(expert_threshold=config.prune_threshold, neuron_threshold=config.prune_threshold)
        new_params, new_state_moe = hk.transform_with_state(lambda x: new_moe(x)).init(rng, inputs[0])
        params["moe"] = new_params
        state["moe"] = new_state_moe

        transformer = new_state["transformer"]
        new_transformer = transformer.prune_layers(threshold=config.prune_threshold)
        new_params, new_state_tf = hk.transform_with_state(
                                        lambda x: new_transformer(x, return_attention=True)
                                    ).init(rng, inputs[0])
        params["transformer"] = new_params
        state["transformer"] = new_state_tf

        return params, state

    def meta_step(params, state, meta_opt_state, inner_opt_state, rng, support_inputs, support_modality_types, 
                  support_targets, support_output_modality, query_inputs, query_modality_types, query_targets, 
                  query_output_modality, accum_steps=2, prune_interval=50):
        support_modality_types = tuple(support_modality_types)
        query_modality_types = tuple(query_modality_types)
        adapted_params = params
        adapted_state = state
        new_inner_opt_state = inner_opt_state

        for _ in range(config.num_inner_steps):
            loss, adapted_state, support_thoughts, new_inner_opt_state = train_step(
                adapted_params, adapted_state, new_inner_opt_state, rng, support_inputs, support_modality_types,
                support_targets, support_output_modality, 0, accum_steps=accum_steps, prune_interval=prune_interval
            )
            adapted_params = jax.tree.map(lambda p: p, adapted_params)

        query_embeddings = jnp.mean(get_embeddings(config, params, adapted_state, rng, query_inputs, query_modality_types,
                                                  spike_threshold=config.spike_threshold, epsilon=config.EPSILON), axis=1)
        query_key_np = np.asarray(jax.device_get(query_embeddings), dtype=config.embedding_dtype)
        ltm_memory_query = jnp.array(ltm.retrieve(query_key_np, config.spike_threshold, config.EPSILON), dtype=config.embedding_dtype)
        stm_memory_query = jnp.array(stm.retrieve(query_key_np, config.spike_threshold, config.EPSILON), dtype=config.embedding_dtype)
        mtm_memory_query = jnp.array(mtm.retrieve(query_key_np, config.spike_threshold, config.EPSILON), dtype=config.embedding_dtype)

        loss, new_state, query_thoughts, metrics = compute_loss(
            adapted_params, adapted_state, rng, query_inputs, query_modality_types, query_targets, query_output_modality,
            ltm_memory_query, stm_memory_query, mtm_memory_query
        )

        grads = jax.grad(loss_for_gradients)(params, state, rng, query_inputs, query_modality_types, query_targets, 
                                            query_output_modality, ltm_memory_query, stm_memory_query, mtm_memory_query)
        grads = clip_gradients(config, grads, max_norm=10.0)
        updates, new_meta_opt_state = meta_optimizer.update(grads, meta_opt_state, params)
        params = optax.apply_updates(params, updates)

        return loss, params, new_state, new_meta_opt_state, support_thoughts, query_thoughts, new_inner_opt_state

    task_batches = create_multimodal_batches(multimodal_datasets, config.task_size, shuffle=True)
    logger.info(f"Created {len(task_batches)} multimodal task batches")

    for task_idx, (inputs, modality_types, targets, output_modality) in enumerate(task_batches):
        support_inputs = [inp[:5] for inp in inputs]
        support_targets = targets[:5] if targets is not None else None
        query_inputs = [inp[5:] for inp in inputs]
        query_targets = targets[5:] if targets is not None else None

        if any(inp.shape[0] < 5 for inp in support_inputs) or any(inp.shape[0] < 1 for inp in query_inputs):
            logger.info(f"Skipping task {task_idx+1} - insufficient support/query samples")
            continue

        logger.info(f"Processing task {task_idx+1}/{len(task_batches)} ({(task_idx + 1) / len(task_batches) * 100:.1f}% complete)")
        step_rng, rng = jax.random.split(rng)
        try:
            loss, params, state, meta_opt_state, support_thoughts, query_thoughts, inner_opt_state = meta_step(
                params, state, meta_opt_state, inner_opt_state, step_rng, 
                support_inputs, modality_types, support_targets, output_modality,
                query_inputs, modality_types, query_targets, output_modality,
                accum_steps=2, prune_interval=config.prune_interval
            )
        except Exception as e:
            logger.error(f"[Task {task_idx+1}] Failed: {e}")
            continue

        losses.append(float(jax.device_get(loss)))
        embeddings = jnp.mean(get_embeddings(config, params, state, step_rng, query_inputs, modality_types,
                                            spike_threshold=config.spike_threshold, epsilon=config.EPSILON), axis=1)
        ltm.store(embeddings, embeddings, config.spike_threshold, config.EPSILON)
        stm.store(embeddings, embeddings, config.spike_threshold, config.EPSILON)
        mtm.store(embeddings, embeddings, config.spike_threshold, config.EPSILON)

        thought_logs.append({"task_idx": task_idx, "support_thoughts": support_thoughts, "query_thoughts": query_thoughts})
        similarity_score = thought_logs[-1]["query_thoughts"]["similarity_score"]
        similarity_score_mean = float(jax.device_get(jnp.mean(similarity_score)))
        similarity_scores.append(similarity_score_mean)

        logger.info(f"[Task {task_idx+1}] Loss: {losses[-1]:.4f} | Similarity: {similarity_score_mean:.4f}")
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

def clip_gradients(config, grads, max_norm=10.0):
    """Clip gradients by their norm to prevent exploding gradients."""
    def clip_by_norm(grad):
        if grad.size == 0:
            return grad
        grad_norm = jnp.linalg.norm(grad)
        scale = jnp.minimum(1.0, max_norm / (grad_norm + config.EPSILON))
        return grad * scale
    return jax.tree.map(clip_by_norm, grads)

def get_embeddings(config, params, state, rng, inputs, modality_types, retrieved_memory_ltm=None, 
                   retrieved_memory_stm=None, retrieved_memory_mtm=None, spike_threshold=0.1, epsilon=1e-8):
    modality_types = tuple(modality_types)
    def forward_with_embeddings(inputs, modality_types, output_modality="text", return_attention=False, 
                                retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None, 
                                spike_threshold=spike_threshold, epsilon=epsilon):
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
            mtm_weight=config.mtm_weight,
            audio_sample_rate=config.audio_sample_rate,
            image_size=config.image_size
        )
        x = jnp.zeros((inputs[0].shape[0], config.max_seq_length, config.d_model)) # Initialize with zeros
        for inp, m_type in zip(inputs, modality_types):
            emb = model.encode_modality(inp, m_type)
            if emb.shape[1] > config.max_seq_length:
                emb = emb[:, :config.max_seq_length, :]
            elif emb.shape[1] < config.max_seq_length:
                padding = jnp.zeros((emb.shape[0], config.max_seq_length - emb.shape[1], config.d_model))
                emb = jnp.concatenate([emb, padding], axis=1)
            x += emb  # Simple addition for initialization (fusion handled in model)
        x += model.position_enc(jnp.arange(config.max_seq_length))
        if retrieved_memory_ltm is not None:
            ltm_emb = model.memory_projection_ltm(retrieved_memory_ltm[:, None, :])
            ltm_emb = jnp.repeat(ltm_emb, config.max_seq_length, axis=1)
            x += model.ltm_weight * ltm_emb
        if retrieved_memory_stm is not None:
            stm_emb = model.memory_projection_stm(retrieved_memory_stm[:, None, :])
            stm_emb = jnp.repeat(stm_emb, config.max_seq_length, axis=1)
            x += model.stm_weight * stm_emb
        if retrieved_memory_mtm is not None:
            mtm_emb = model.memory_projection_mtm(retrieved_memory_mtm[:, None, :])
            mtm_emb = jnp.repeat(mtm_emb, config.max_seq_length, axis=1)
            x += model.mtm_weight * mtm_emb
        if "text" in modality_types and modality_types[0] == "text":
            x, _ = model.self_attention(inputs[0], return_attention=True, spike_threshold=spike_threshold, epsilon=epsilon)
        x, _ = model.transformer(x, rng, return_attention=True, spike_threshold=spike_threshold, epsilon=epsilon)
        x, _, _ = model.moe(x, spike_threshold=spike_threshold, epsilon=epsilon)
        x = model.norm(x)
        return x

    transformed = hk.transform_with_state(forward_with_embeddings)
    embeddings, _ = transformed.apply(params, state, rng, inputs, modality_types, "text",
                                     retrieved_memory_ltm=retrieved_memory_ltm,
                                     retrieved_memory_stm=retrieved_memory_stm,
                                     retrieved_memory_mtm=retrieved_memory_mtm)
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
        prune_threshold=0.01,
        prune_interval=50,
        audio_sample_rate=16000,
        image_size=64
    )
    losses, params, similarity_scores, state, ltm, stm, mtm, thought_logs = train_and_evaluate(config, [], [], [])
    logger.info(f"Training completed - Final Loss: {losses[-1]:.4f}, Final Similarity: {similarity_scores[-1]:.4f}")