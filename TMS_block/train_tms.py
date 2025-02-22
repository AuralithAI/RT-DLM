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
from data_utils import DataProcessor, load_data, create_batches
from memory_bank import MemoryBank, ShortTermMemory, MidTermMemory

def train_and_evaluate(config, losses, similarity_scores, thought_logs):
    rng = jax.random.PRNGKey(42)
    devices = jax.devices()  
    print(f"[INFO] Detected devices: {devices}")

    ltm = MemoryBank(memory_size=config.memory_size, embedding_dim=config.d_model, retrieval_k=config.retrieval_k)
    stm = ShortTermMemory(buffer_size=config.stm_buffer_size, embedding_dim=config.d_model)
    mtm = MidTermMemory(buffer_size=config.mtm_buffer_size, embedding_dim=config.d_model, retention_steps=config.retention_steps)

    dataset_path = "data/train_data.txt"
    processor = DataProcessor(vocab_size=config.vocab_size)
    raw_texts = load_data(dataset_path)
    tokenized_texts = [processor.tokenize(text) for text in raw_texts]
    inputs_np = np.array([processor.pad_sequence(tokens, config.max_seq_length) for tokens in tokenized_texts], dtype=np.int32)
    targets_np = inputs_np  

    # Model definition
    def forward_fn(inputs, return_attention=False, retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None):
        model = TMSModel(
            d_model=config.d_model, num_heads=config.num_heads, num_layers=config.num_layers,
            vocab_size=config.vocab_size, max_seq_length=config.max_seq_length,
            moe_experts=config.moe_experts, moe_top_k=config.moe_top_k,
            memory_size=config.memory_size, retrieval_k=config.retrieval_k,
            ltm_weight=config.ltm_weight, stm_weight=config.stm_weight, mtm_weight=config.mtm_weight
        )
        return model(inputs, return_attention=return_attention, retrieved_memory_ltm=retrieved_memory_ltm, 
                     retrieved_memory_stm=retrieved_memory_stm, retrieved_memory_mtm=retrieved_memory_mtm)

    model = hk.transform_with_state(forward_fn)
    inputs = jnp.array(inputs_np[:config.batch_size], dtype=jnp.int32)  
    params, state = model.init(rng, inputs)  

    # Meta-learning optimizers
    meta_optimizer = optax.adam(config.learning_rate)
    inner_optimizer = optax.sgd(config.inner_learning_rate)
    meta_opt_state = meta_optimizer.init(params)

    # Loss function with thought tracking
    def compute_loss(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm):
        (logits, (attn_weights_self, attn_weights_transformer), expert_indices, aux_loss), new_state = model.apply(
            params, state, rng, inputs, return_attention=True,
            retrieved_memory_ltm=retrieved_memory_ltm, retrieved_memory_stm=retrieved_memory_stm, retrieved_memory_mtm=retrieved_memory_mtm
        )
        embeddings = get_embeddings(config, params, state, rng, inputs, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm)
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

        thoughts = {
            "attn_weights_self": attn_weights_self,
            "attn_weights_transformer": attn_weights_transformer,
            "similarity_score": similarity_score,
            "expert_indices": expert_indices
        }

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        total_loss = loss + 0.01 * aux_loss
        return total_loss, attn_weights_self, attn_weights_transformer, expert_indices, similarity_score, new_state, thoughts

    # Loss for gradients
    def loss_for_gradients(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm):
        loss, _, _, _, _, _, _ = compute_loss(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm)
        return loss

    # Inner loop train step
    def train_step(params, state, opt_state, rng, inputs, targets, step, ltm_memory_support, stm_memory_support, mtm_memory_support):
        embeddings = jnp.mean(get_embeddings(config, params, state, rng, inputs, ltm_memory_support, stm_memory_support, mtm_memory_support), axis=1)
        query_key_np = np.asarray(jax.device_get(embeddings), dtype=np.float32)
        ltm_memory = jnp.array(ltm.retrieve(query_key_np, config.EPSILON), dtype=jnp.float32)
        stm_memory = jnp.array(stm.retrieve(query_key_np, config.EPSILON), dtype=jnp.float32)
        mtm_memory = jnp.array(mtm.retrieve(query_key_np, config.EPSILON), dtype=jnp.float32)
        
        loss, attn_weights_self, attn_weights_transformer, expert_indices, similarity_score, new_state, thoughts = _train_step_jit(
            params, state, opt_state, rng, inputs, targets, ltm_memory, stm_memory, mtm_memory
        )
        return loss, new_state, thoughts

    @jax.jit
    def _train_step_jit(params, state, opt_state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm):
        loss, attn_weights_self, attn_weights_transformer, expert_indices, similarity_score, new_state, thoughts = compute_loss(
            params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm
        )
        grads = jax.grad(loss_for_gradients)(params, state, rng, inputs, targets, retrieved_memory_ltm, retrieved_memory_stm, retrieved_memory_mtm)
        updates, opt_state = inner_optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, attn_weights_self, attn_weights_transformer, expert_indices, similarity_score, new_state, thoughts

    # Meta-step: Outer loop
    @jax.jit
    def meta_step(params, state, meta_opt_state, rng, support_inputs, support_targets, query_inputs, query_targets):
        # Inner loop: Adapt to support set with memory support
        support_embeddings = jnp.mean(get_embeddings(config, params, state, rng, support_inputs), axis=1)
        query_key_np_support = np.asarray(jax.device_get(support_embeddings), dtype=np.float32)
        ltm_memory_support = jnp.array(ltm.retrieve(query_key_np_support, config.EPSILON), dtype=jnp.float32)
        stm_memory_support = jnp.array(stm.retrieve(query_key_np_support, config.EPSILON), dtype=jnp.float32)
        mtm_memory_support = jnp.array(mtm.retrieve(query_key_np_support, config.EPSILON), dtype=jnp.float32)
        
        loss, adapted_state, support_thoughts = train_step(params, state, None, rng, support_inputs, support_targets, 0, 
                                                          ltm_memory_support, stm_memory_support, mtm_memory_support)
        adapted_params = jax.tree_map(lambda p: p, params)

        # Outer loop: Evaluate on query set
        query_embeddings = jnp.mean(get_embeddings(config, adapted_params, adapted_state, rng, query_inputs), axis=1)
        query_key_np_query = np.asarray(jax.device_get(query_embeddings), dtype=np.float32)
        ltm_memory_query = jnp.array(ltm.retrieve(query_key_np_query, config.EPSILON), dtype=jnp.float32)
        stm_memory_query = jnp.array(stm.retrieve(query_key_np_query, config.EPSILON), dtype=jnp.float32)
        mtm_memory_query = jnp.array(mtm.retrieve(query_key_np_query, config.EPSILON), dtype=jnp.float32)
        
        loss, _, _, _, similarity_score, new_state, query_thoughts = compute_loss(
            adapted_params, adapted_state, rng, query_inputs, query_targets, ltm_memory_query, stm_memory_query, mtm_memory_query
        )
        
        grads = jax.grad(loss_for_gradients)(params, state, rng, query_inputs, query_targets, ltm_memory_query, stm_memory_query, mtm_memory_query)
        updates, meta_opt_state = meta_optimizer.update(grads, meta_opt_state)
        params = optax.apply_updates(params, updates)
        
        return loss, params, new_state, meta_opt_state, support_thoughts, query_thoughts, similarity_score

    # Use create_batches to generate task-sized batches
    task_size = config.task_size
    num_tasks = len(inputs_np) // task_size
    task_batches = create_batches(jnp.array(inputs_np, dtype=jnp.int32), jnp.array(targets_np, dtype=jnp.int32), task_size, shuffle=True)
    
    for task_idx, (task_inputs, task_targets) in enumerate(task_batches):
        if task_idx >= min(num_tasks, config.num_epochs * 100):
            break
        
        support_inputs = task_inputs[:5]
        support_targets = task_targets[:5]
        query_inputs = task_inputs[5:]
        query_targets = task_targets[5:]

        if len(support_inputs) < 5 or len(query_inputs) < 1:
            continue

        step_rng, rng = jax.random.split(rng)
        loss, params, state, meta_opt_state, support_thoughts, query_thoughts, similarity_score = meta_step(
            params, state, meta_opt_state, step_rng, support_inputs, support_targets, query_inputs, query_targets
        )
        losses.append(float(loss))

        embeddings = jnp.mean(get_embeddings(config, params, state, step_rng, query_inputs), axis=1)
        embeddings_np = np.asarray(jax.device_get(embeddings), dtype=np.float32)
        ltm.store(embeddings_np, embeddings_np)
        stm.store(embeddings_np, embeddings_np)
        mtm.store(embeddings_np, embeddings_np)

        thought_logs.append({"task_idx": task_idx, "support_thoughts": support_thoughts, "query_thoughts": query_thoughts})
        similarity_score_mean = np.mean(np.array(similarity_score).astype(float))
        similarity_scores.append(similarity_score_mean)

        print(f"[Task {task_idx+1}] Loss: {loss:.4f} | Similarity: {similarity_score_mean:.4f}")
        gc.collect()
        jax.clear_caches()

    return losses, params, similarity_scores, state, ltm, stm, mtm, thought_logs

def get_embeddings(config, params, state, rng, inputs, retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None):
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