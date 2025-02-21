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
from memory_bank import MemoryBank, ShortTermMemory

def train_and_evaluate(config, losses, similarity_scores):
    rng = jax.random.PRNGKey(42)
    ltm = MemoryBank(memory_size=config.memory_size, embedding_dim=config.d_model, retrieval_k=config.retrieval_k)
    stm = ShortTermMemory(buffer_size=config.buffer_size, embedding_dim=config.d_model)

    # Load dataset
    dataset_path = "data/train_data.txt"
    processor = DataProcessor(vocab_size=config.vocab_size)
    raw_texts = load_data(dataset_path)

    tokenized_texts = [processor.tokenize(text) for text in raw_texts]
    inputs = jnp.array([processor.pad_sequence(tokens, config.max_seq_length) for tokens in tokenized_texts], dtype=jnp.int32)
    targets = jnp.array(inputs, dtype=jnp.int32)

    # Initialize TMS Model
    def forward_fn(inputs, return_attention=False, retrieved_memory_ltm=None, retrieved_memory_stm=None):
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
            stm_weight=config.stm_weight
        )
        return model(inputs, return_attention=return_attention, retrieved_memory_ltm=retrieved_memory_ltm, retrieved_memory_stm=retrieved_memory_stm)

    model = hk.transform_with_state(forward_fn)
    params, state = model.init(rng, inputs[:config.batch_size])

    # Optimizer
    warmup_steps = 5000
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=2e-6,
        peak_value=config.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=200000,
        end_value=2e-6
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(schedule, weight_decay=1e-3)
    )
    opt_state = optimizer.init(params)

    # Compute loss function
    def compute_loss(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm):
        (logits, attn_weights, expert_indices, aux_loss), new_state = model.apply(
            params, state, rng, inputs, return_attention=True, 
            retrieved_memory_ltm=retrieved_memory_ltm, retrieved_memory_stm=retrieved_memory_stm
        )
        ltm_norm = jnp.linalg.norm(retrieved_memory_ltm, axis=-1, keepdims=True) + config.EPSILON
        stm_norm = jnp.linalg.norm(retrieved_memory_stm, axis=-1, keepdims=True) + config.EPSILON
        retrieved_memory_ltm = retrieved_memory_ltm / ltm_norm
        retrieved_memory_stm = retrieved_memory_stm / stm_norm

        embeddings = get_embeddings(params, state, rng, inputs, retrieved_memory_ltm, retrieved_memory_stm)
        query_key = jnp.mean(embeddings, axis=1)
        query_norm = jnp.linalg.norm(query_key, axis=-1, keepdims=True) + config.EPSILON
        ltm_similarity = jnp.sum(query_key * retrieved_memory_ltm, axis=-1) / (query_norm * ltm_norm)
        stm_similarity = jnp.sum(query_key * retrieved_memory_stm, axis=-1) / (query_norm * stm_norm)
        similarity_score = jnp.stack([ltm_similarity, stm_similarity], axis=-1)  # [batch_size, 2]
        similarity_score = jax.lax.stop_gradient(similarity_score)
        similarity_score = jnp.nan_to_num(similarity_score, nan=0.0)

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        total_loss = loss + 0.01 * aux_loss
        return total_loss, attn_weights, expert_indices, similarity_score, new_state

    def loss_for_gradients(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm):
        loss, _, _, _, _ = compute_loss(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm)
        return loss

    def train_step(params, state, opt_state, rng, inputs, targets):
        embeddings = jnp.mean(get_embeddings(params, state, rng, inputs), axis=1)
        query_key_np = np.asarray(jax.device_get(embeddings), dtype=np.float32)
        ltm_memory_np = ltm.retrieve(query_key_np)
        stm_memory_np = stm.retrieve(query_key_np)
        ltm_memory = jnp.array(ltm_memory_np, dtype=jnp.float32)
        stm_memory = jnp.array(stm_memory_np, dtype=jnp.float32)
        return _train_step_jit(params, state, opt_state, rng, inputs, targets, ltm_memory, stm_memory)

    @jax.jit
    def _train_step_jit(params, state, opt_state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm):
        loss, attn_weights, expert_indices, similarity_score, new_state = compute_loss(
            params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm
        )
        grads = jax.grad(loss_for_gradients)(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, attn_weights, expert_indices, params, state, opt_state, similarity_score

    def get_embeddings(params, state, rng, inputs, retrieved_memory_ltm=None, retrieved_memory_stm=None):
        def forward_with_embeddings(inputs, return_attention=False, retrieved_memory_ltm=None, retrieved_memory_stm=None):
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
                stm_weight=config.stm_weight
            )
            x = model.embedding(inputs) + model.position_enc(jnp.arange(inputs.shape[1]))
            if retrieved_memory_ltm is not None:
                retrieved_memory_ltm = jnp.expand_dims(retrieved_memory_ltm, axis=1)
                retrieved_memory_ltm = jnp.repeat(retrieved_memory_ltm, x.shape[1], axis=1)
                retrieved_memory_ltm = model.memory_projection_ltm(retrieved_memory_ltm)
                x += model.ltm_weight * retrieved_memory_ltm
            if retrieved_memory_stm is not None:
                retrieved_memory_stm = jnp.expand_dims(retrieved_memory_stm, axis=1)
                retrieved_memory_stm = jnp.repeat(retrieved_memory_stm, x.shape[1], axis=1)
                retrieved_memory_stm = model.memory_projection_stm(retrieved_memory_stm)
                x += model.stm_weight * retrieved_memory_stm
            x, _ = model.self_attention(inputs, return_attention=True)
            x, _ = model.transformer(x, rng, return_attention=True)
            x, _, _ = model.moe(x)
            x = model.norm(x)
            return x

        transformed = hk.transform_with_state(forward_with_embeddings)
        embeddings, _ = transformed.apply(params, state, rng, inputs, retrieved_memory_ltm=retrieved_memory_ltm, retrieved_memory_stm=retrieved_memory_stm)
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
            loss, _, _, params, state, opt_state, similarity_score = train_step(params, state, opt_state, step_rng, batch_inputs, batch_targets)
            losses.append(loss)

            embeddings = jnp.mean(get_embeddings(params, state, step_rng, batch_inputs), axis=1)
            ltm.store(embeddings, embeddings)
            stm.store(embeddings, embeddings)

            similarity_score_numpy = np.array(similarity_score).astype(float)
            similarity_score_mean = np.mean(similarity_score_numpy)
            similarity_scores.append(similarity_score_mean)

            print(f"[Epoch {epoch+1} | Step {step+1}] Loss: {loss:.4f} | Similarity Score: {similarity_score_mean:.4f}")
            del batch_inputs, batch_targets
            gc.collect()

    return losses, params, similarity_scores, state, ltm