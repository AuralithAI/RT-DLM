import jax
import jax.numpy as jnp
import haiku as hk
import optax
import os
from model_module import RTDLMModel
from train_config import TrainConfig
from data_utils import DataProcessor, load_data, preprocess_batch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "gpu")

def visualize_embeddings(embeddings, step):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    plt.title(f"Embedding Space at Step {step}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()
    plt.savefig(f"embedding_step_{step}.png")
    plt.close()

def forward_fn(inputs):
    config = TrainConfig()
    model = RTDLMModel(config)
    return model(inputs)

# ✅ Haiku expects functions to be transformed this way
model = hk.transform_with_state(forward_fn)
optimizer = optax.adamw(TrainConfig().learning_rate)

@jax.jit
def update(params, state, opt_state, inputs, targets):
    def loss_fn(params, state):
        predictions, new_state = model.apply(params, state, hk.next_rng_key(), inputs)  # ✅ Fix: use Haiku's PRNG
        loss = jnp.mean(optax.softmax_cross_entropy(predictions, jax.nn.one_hot(targets, TrainConfig().vocab_size)))
        return loss, new_state  

    (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    
    return loss, new_params, new_state, opt_state

def train():
    config = TrainConfig()
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

    dummy_inputs, _ = next(data_generator(train_data, config.batch_size))
    print(f"Dummy Inputs Shape: {dummy_inputs.shape}")
    assert dummy_inputs.shape == (config.batch_size, config.max_seq_length), \
        f"Expected shape: {(config.batch_size, config.max_seq_length)}, got {dummy_inputs.shape}"
    
    # ✅ Corrected model initialization
    params, state = model.init(rng, dummy_inputs)
    opt_state = optimizer.init(params)

    losses = []

    for epoch in range(config.num_epochs):
        print(f"[Training] Starting Epoch {epoch + 1}")
        for step, (inputs, targets) in enumerate(data_generator(train_data, config.batch_size)):
            print(f"[Training] Step {step}, Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
            
            loss, params, state, opt_state = update(params, state, opt_state, inputs, targets)
            print(f"[Training] Step {step}, Loss: {loss:.4f}")

            if step % config.eval_interval == 0:
                embeddings, state = model.apply(params, state, hk.next_rng_key(), dummy_inputs)
                print(f"[Training] Embeddings shape: {embeddings.shape}")
                visualize_embeddings(embeddings, step)

    print("Training complete!")

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("training_loss_plot.png")
    plt.close()

if __name__ == "__main__":
    train()