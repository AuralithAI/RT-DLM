import jax
import jax.numpy as jnp
import haiku as hk
import optax
import os
from model_module import model  
from train_config import TrainConfig
from data_utils import DataProcessor, load_data, preprocess_batch

jax.config.update("jax_platform_name", "gpu")
MAX_GRAD_NORM = 1.0 # Prevent exploding gradients

config = TrainConfig()
optimizer = optax.adamw(config.learning_rate)

@jax.jit
def update(params, state, opt_state, rng, inputs, targets):
    def loss_fn(params, state, rng, targets):
        rng, subkey = jax.random.split(rng)  
        predictions, new_state = model.apply(params, state, subkey, inputs)

        if jnp.isnan(predictions).any():
            print("[ERROR] NaN detected in predictions!")

        targets_one_hot = jax.nn.one_hot(targets, config.vocab_size)

        if jnp.isnan(targets_one_hot).any():
            print("[ERROR] NaN detected in one-hot targets!")

        # 🔥 Apply `log_softmax` to prevent instability
        log_probs = jax.nn.log_softmax(predictions, axis=-1)
        loss = -jnp.sum(targets_one_hot * log_probs) / targets.shape[1]

        if jnp.isnan(loss):
            print("[ERROR] NaN detected in loss!")

        return loss, new_state  

    (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, rng, targets)

    if jnp.isnan(grads).any():
        print("[ERROR] NaN detected in gradients!")

    grads = jax.tree_map(lambda g: jnp.clip(g, -MAX_GRAD_NORM, MAX_GRAD_NORM), grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)  
    new_params = optax.apply_updates(params, updates)

    return loss, new_params, new_state, opt_state

def train():
    processor = DataProcessor()
    data = load_data(os.path.join(os.getcwd(), f"data{os.sep}dataset.txt"))
    processor.build_vocab(data)
    train_data, val_data = data[:int(0.9 * len(data))], data[int(0.9 * len(data)):]

    def data_generator(data, batch_size):
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            if len(batch) < batch_size:
                pad_count = batch_size - len(batch)
                batch.extend([""] * pad_count) 
            inputs, targets = preprocess_batch(batch, processor, config.max_seq_length)
            yield inputs, targets

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    dummy_inputs = jax.random.randint(init_rng, shape=(config.batch_size, config.max_seq_length), minval=0, maxval=config.vocab_size)
    print(f"Dummy Inputs Shape: {dummy_inputs.shape}")

    params, state = model.init(init_rng, dummy_inputs, init_rng)
    opt_state = optimizer.init(params)

    for epoch in range(config.num_epochs):
        print(f"[Training] Starting Epoch {epoch + 1}")
        for step, (inputs, targets) in enumerate(data_generator(train_data, config.batch_size)):
            print(f"[DEBUG] Step {step}: inputs.shape={inputs.shape}, targets.shape={targets.shape}")

            if inputs.shape[1] != config.max_seq_length:
                print(f"[DEBUG] Fixing inputs shape: {inputs.shape} → (1, {config.max_seq_length})")
                pad_width = config.max_seq_length - inputs.shape[1]
                inputs = jnp.pad(inputs, ((0, 0), (0, pad_width)), constant_values=0)

            rng, step_rng = jax.random.split(rng)
            loss, params, state, opt_state = update(params, state, opt_state, step_rng, inputs, targets)
            print(f"[Training] Step {step}, Loss: {loss:.4f}")

    print("Training complete!")

if __name__ == "__main__":
    train()