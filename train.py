import jax
import jax.numpy as jnp
import optax
import os
import matplotlib.pyplot as plt
import pickle
from model_module import model  
from train_config import TrainConfig
from data_utils import DataProcessor, load_data, preprocess_batch, fetch_wikipedia_data, fetch_commoncrawl_data, save_dataset

jax.config.update("jax_platform_name", "gpu")
file_path = os.path.join(os.getcwd(), "data/dataset.txt")
MAX_GRAD_NORM = 1.0

@jax.jit
def update(params, state, opt_state, rng, inputs, targets):
    def loss_fn(params, state, rng, targets):
        rng, subkey = jax.random.split(rng)
        predictions, load_balancing_loss = model.apply(params, state, subkey, inputs)

        print(f"[DEBUG] Predictions Type: {type(predictions)}, Shape: {getattr(predictions, 'shape', 'Unknown')}")
        print(f"[DEBUG] Load Balancing Loss Type: {type(load_balancing_loss)}, Value: {load_balancing_loss}")

        if isinstance(predictions, tuple): 
            predictions = predictions[0]

        if isinstance(load_balancing_loss, dict): 
            load_balancing_loss = jnp.array(0.0)

        targets_one_hot = jax.nn.one_hot(targets, config.vocab_size)
        log_probs = jax.nn.log_softmax(predictions, axis=-1)
        
        task_loss = -jnp.mean(jnp.sum(targets_one_hot * log_probs, axis=-1))
        total_loss = task_loss + 0.01 * load_balancing_loss

        return total_loss, load_balancing_loss

    (loss, load_balancing_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, rng, targets)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return loss, load_balancing_loss, new_params, opt_state

def train():
    data = load_data(file_path=file_path)
    processor.build_vocab(data)

    train_data, val_data = data[:int(0.9 * len(data))], data[int(0.9 * len(data)):]

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    dummy_inputs = jax.random.randint(init_rng, shape=(config.batch_size, config.max_seq_length), minval=0, maxval=config.vocab_size)
    params, state = model.init(init_rng, dummy_inputs, init_rng)
    opt_state = optimizer.init(params)

    loss_history = []

    for epoch in range(config.num_epochs):
        print(f"[Training] Epoch {epoch + 1}")

        for step, (inputs, targets) in enumerate(data_generator(train_data, config.batch_size)):
            rng, step_rng = jax.random.split(rng)
            loss, load_balancing_loss, params, opt_state = update(params, state, opt_state, step_rng, inputs, targets)
            loss_history.append(loss)
            print(f"[Step {step}] Loss: {loss:.4f}, MoE Load Balancing Loss: {load_balancing_loss:.4f}")

    with open("rt_dlm_model.pkl", "wb") as f:
        pickle.dump(params, f)
        print("[INFO] Model parameters saved to rt_dlm_model.pkl")

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid()
    plt.savefig("loss_plot.png")
    plt.show()
    print("[INFO] Loss plot saved as loss_plot.png")
    print("Training complete!")

def data_generator(data, batch_size):
    """
    Generator function that yields batches of tokenized and padded input-target pairs.

    Args:
        data (List[str]): List of text samples from the dataset.
        batch_size (int): Number of samples per batch.

    Yields:
        Tuple[jnp.ndarray, jnp.ndarray]: Tokenized and padded input-target tensors.
    """
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        if len(batch) < batch_size:
            pad_count = batch_size - len(batch)
            batch.extend([""] * pad_count)

        inputs, targets = preprocess_batch(batch, processor, config.max_seq_length)
        
        yield inputs, targets

if __name__ == "__main__":
    processor = DataProcessor()
    config = TrainConfig()
    optimizer = optax.adamw(config.learning_rate)
    if not os.path.exists(file_path):
        wiki_data = fetch_wikipedia_data(num_articles=1000)
        #crawl_data = fetch_commoncrawl_data()
        all_data = wiki_data #+ crawl_data
        all_data = [processor.preprocess_text(text) for text in all_data]
        save_dataset(all_data)
    else:
        print("[INFO] Dataset exists from sources.")
    train()