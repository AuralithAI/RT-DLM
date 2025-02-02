import jax
import jax.numpy as jnp
import haiku as hk
import optax
import os
from model import EmbeddingLayer, TransformerBlock, MixtureOfExperts, to_device
from config import RTDLMConfig
from data_utils import DataProcessor, load_data, preprocess_batch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

jax.config.update("jax_disable_jit", True)
jax.config.update("jax_platform_name", "gpu")

class TrainConfig:
    def __init__(self):
        self.vocab_size = 4000
        self.d_model = 64
        self.num_heads = 2
        self.num_layers = 2
        self.moe_experts = 2
        self.moe_top_k = 1
        self.max_seq_length = 64
        self.batch_size = 1
        self.learning_rate = 3e-4
        self.num_epochs = 3
        self.eval_interval = 50

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
                # Pad batch to batch_size
                pad_count = batch_size - len(batch)
                batch.extend([""] * pad_count)  # Assuming empty strings for padding
            inputs, targets = preprocess_batch(batch, processor, config.max_seq_length)
            yield inputs, targets


    def forward_fn(inputs):
        embedding_layer = EmbeddingLayer(config.vocab_size, config.d_model, config.max_seq_length)
        x = embedding_layer(inputs, inputs.shape[1])

        for _ in range(config.num_layers):
            transformer_block = TransformerBlock(config.d_model, config.num_heads)
            x = transformer_block(x)

        moe_layer = MixtureOfExperts(config.d_model, config.moe_experts, config.moe_top_k, dropout_rate=0.1)
        x = moe_layer(x, is_training=True)

        return to_device(x)

    model = hk.transform_with_state(forward_fn)
    optimizer = optax.adamw(config.learning_rate)

    @jax.jit
    def update(params, opt_state, inputs, targets):
        def loss_fn(params):
            predictions = model.apply(params, None, inputs)
            loss = jnp.mean(optax.softmax_cross_entropy(predictions, jax.nn.one_hot(targets, config.vocab_size)))
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, opt_state

    rng = jax.random.PRNGKey(42)
    dummy_inputs, _ = next(data_generator(train_data, config.batch_size))
    assert dummy_inputs.shape == (config.batch_size, config.max_seq_length), \
    f"Expected shape: {(config.batch_size, config.max_seq_length)}, got {dummy_inputs.shape}"
    params = model.init(rng, dummy_inputs)
    opt_state = optimizer.init(params)

    losses = []

    for epoch in range(config.num_epochs):
        print(f"[Training] Starting Epoch {epoch + 1}")
        for step, (inputs, targets) in enumerate(data_generator(train_data, config.batch_size)):
            print(f"[Training] Step {step}, Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
            
            loss, params, opt_state = update(params, opt_state, inputs, targets)
            print(f"[Training] Step {step}, Loss: {loss:.4f}")

            if step % config.eval_interval == 0:
                embeddings = model.apply(params, None, dummy_inputs)
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

if __name__ == "__main__":
    train()
