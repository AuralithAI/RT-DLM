import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import gc
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_tms import TMSModel
from data_utils import DataProcessor, load_data
from memory_bank import MemoryBank, ShortTermMemory, MidTermMemory

def train_and_evaluate(config, losses, similarity_scores):
    rng = jax.random.PRNGKey(42)
    ltm = MemoryBank(memory_size=config.memory_size, embedding_dim=config.d_model, retrieval_k=config.retrieval_k)
    stm = ShortTermMemory(buffer_size=config.stm_buffer_size, embedding_dim=config.d_model)
    mtm = MidTermMemory(buffer_size=config.mtm_buffer_size, embedding_dim=config.d_model, retention_steps=config.retention_steps)

    dataset_path = "data/train_data.txt"
    processor = DataProcessor(vocab_size=config.vocab_size)
    raw_texts = load_data(dataset_path)
    tokenized_texts = [processor.tokenize(text) for text in raw_texts]
    inputs = jnp.array([processor.pad_sequence(tokens, config.max_seq_length) for tokens in tokenized_texts], dtype=jnp.int32)
    targets = jnp.array(inputs, dtype=jnp.int32)

    def forward_fn(inputs, return_attention=False, retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None):
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
                     retrieved_memory_stm=retrieved_memory_stm, retrieved_memory_mtm=retrieved_memory_mtm)

    model = hk.transform_with_state(forward_fn)
    params, state = model.init(rng, inputs[:config.batch_size])

    # Optimizer with config-based parameters
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.init_lr,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.decay_steps,
        end_value=config.end_lr
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.clip_norm),
        optax.adamw(schedule, weight_decay=config.weight_decay)
    )
    opt_state = optimizer.init(params)

    def compute_loss(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm):
        (logits, attn_weights, expert_indices, aux_loss), new_state = model.apply(
            params, state, rng, inputs, return_attention=True,
            retrieved_memory_ltm=retrieved_memory_ltm, retrieved_memory_stm=retrieved_memory_stm, retrieved_memory_mtm=retrieved_memory_mtm
        )
        embeddings = get_embeddings(params, state, rng, inputs, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm)
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
        similarity_score = jnp.nan_to_num(similarity_score, nan=0.0)

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        total_loss = loss + 0.01 * aux_loss
        return total_loss, attn_weights, expert_indices, similarity_score, new_state

    def loss_for_gradients(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm):
        loss, _, _, _, _ = compute_loss(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm)
        return loss

    def train_step(params, state, opt_state, rng, inputs, targets, step):
        embeddings = jnp.mean(get_embeddings(params, state, rng, inputs), axis=1)
        query_key_np = np.asarray(jax.device_get(embeddings), dtype=np.float32)
        ltm_memory = jnp.array(ltm.retrieve(query_key_np, config.EPSILON), dtype=jnp.float32)
        stm_memory = jnp.array(stm.retrieve(query_key_np, config.EPSILON), dtype=jnp.float32)
        mtm_memory = jnp.array(mtm.retrieve(query_key_np, config.EPSILON), dtype=jnp.float32)
        
        # Self-tuning learning rate (adjust based on loss trend)
        loss, _, _, params, state, opt_state, similarity_score = _train_step_jit(
            params, state, opt_state, rng, inputs, targets, ltm_memory, stm_memory, mtm_memory
        )
        if step > config.warmup_steps and step % config.eval_interval == 0:
            recent_losses = losses[-config.eval_interval:] if len(losses) >= config.eval_interval else losses
            loss_trend = np.mean(recent_losses[-5:]) - np.mean(recent_losses[:5]) if len(recent_losses) >= 5 else 0
            if loss_trend > 0:  
                config.learning_rate *= 0.9  
            elif loss_trend < -0.1:  
                config.learning_rate *= 1.1  
            config.learning_rate = max(min(config.learning_rate, 1e-3), 1e-5)  
        
        return loss, params, state, opt_state, similarity_score

    @jax.jit
    def _train_step_jit(params, state, opt_state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm):
        loss, attn_weights, expert_indices, similarity_score, new_state = compute_loss(
            params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm
        )
        grads = jax.grad(loss_for_gradients)(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, attn_weights, expert_indices, params, state, opt_state, similarity_score

    def get_embeddings(params, state, rng, inputs, retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None):
        def forward_with_embeddings(inputs, return_attention=False, retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None):
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
            x, _ = model.self_attention(inputs, return_attention=True)
            x, _ = model.transformer(x, rng, return_attention=True)
            x, _, _ = model.moe(x)
            x = model.norm(x)
            return x

        transformed = hk.transform_with_state(forward_with_embeddings)
        embeddings, _ = transformed.apply(params, state, rng, inputs, retrieved_memory_ltm=retrieved_memory_ltm, 
                                         retrieved_memory_stm=retrieved_memory_stm, retrieved_memory_mtm=retrieved_memory_mtm)
        return embeddings

    # Training loop
    for epoch in range(config.num_epochs):
        gc.collect()
        jax.clear_caches()
        for step in range(len(inputs) // config.batch_size):
            batch_start = step * config.batch_size
            batch_end = batch_start + config.batch_size
            batch_inputs, batch_targets = inputs[batch_start:batch_end], targets[batch_start:batch_end]

            step_rng, rng = jax.random.split(rng)
            loss, params, state, opt_state, similarity_score = train_step(params, state, opt_state, step_rng, batch_inputs, batch_targets, step)
            losses.append(loss)

            embeddings = jnp.mean(get_embeddings(params, state, step_rng, batch_inputs), axis=1)
            ltm.store(embeddings, embeddings)
            stm.store(embeddings, embeddings)
            mtm.store(embeddings, embeddings)

            similarity_score_numpy = np.array(similarity_score).astype(float)
            similarity_score_mean = np.mean(similarity_score_numpy)
            similarity_scores.append(similarity_score_mean)

            print(f"[Epoch {epoch+1} | Step {step+1}] Loss: {loss:.4f} | Similarity Score: {similarity_score_mean:.4f} | LR: {config.learning_rate:.6f}")
            del batch_inputs, batch_targets
            gc.collect()

    return losses, params, similarity_scores, state, ltm, stm, mtm