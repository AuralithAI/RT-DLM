import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from train_config import TrainConfig
from model_module import model

def compute_loss(predictions, targets, vocab_size):
    """Compute softmax cross-entropy loss."""
    targets_one_hot = jax.nn.one_hot(targets, vocab_size)
    log_probs = jax.nn.log_softmax(predictions, axis=-1)  
    loss = -jnp.mean(jnp.sum(targets_one_hot * log_probs, axis=-1))
    return jnp.where(jnp.isnan(loss) | jnp.isinf(loss), jnp.array(1e-6), loss)

def update(params, state, opt_state, rng, inputs, targets, optimizer):
    """Compute gradients and update model parameters."""
    def loss_fn(params, state, rng, targets):
        predictions, load_balancing_loss = model.apply(params, state, rng, inputs)
        loss = compute_loss(predictions, targets, TrainConfig().vocab_size)
        return loss, load_balancing_loss  # Returning load balancing loss too

    (loss, load_balancing_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, rng, targets)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return loss, load_balancing_loss, new_params, opt_state

def debug_model():
    config = TrainConfig()
    rng = jax.random.PRNGKey(42)

    # Generate a **valid** test batch (batch_size=4, sequence_length=config.max_seq_length)
    test_inputs = jnp.array([
        [5, 10, 15, 20, 25] + [0] * (config.max_seq_length - 5),
        [30, 35, 40, 45, 50] + [0] * (config.max_seq_length - 5),
        [100, 200, 300, 400, 500] + [0] * (config.max_seq_length - 5),
        [1000, 2000, 3000, 4000, 5000] + [0] * (config.max_seq_length - 5)
    ], dtype=jnp.int32)

    # Create dummy targets for loss computation
    test_targets = jnp.array([
        [10, 15, 20, 25, 30] + [0] * (config.max_seq_length - 5),
        [35, 40, 45, 50, 55] + [0] * (config.max_seq_length - 5),
        [200, 300, 400, 500, 600] + [0] * (config.max_seq_length - 5),
        [2000, 3000, 4000, 5000, 6000] + [0] * (config.max_seq_length - 5)
    ], dtype=jnp.int32)

    print(f"[DEBUG] Static Test Input - Before Model")
    print(f"[DEBUG] Min Token ID: {test_inputs.min()}, Max Token ID: {test_inputs.max()}")

    # Initialize model parameters and optimizer
    params, state = model.init(rng, test_inputs, rng)
    optimizer = optax.adamw(learning_rate=3e-4)
    opt_state = optimizer.init(params)

    num_epochs = 10  # Run multiple epochs
    loss_history = []

    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch + 1}] Running Model...")

        rng, step_rng = jax.random.split(rng)

        # Update Model Parameters
        loss, load_balancing_loss, params, opt_state = update(
            params, state, opt_state, step_rng, test_inputs, test_targets, optimizer
        )

        loss_history.append(loss)

        print(f"[DEBUG] Loss: {loss:.6f}, Load Balancing Loss: {load_balancing_loss}")

    # Plot Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o', linestyle='-', label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model Training Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()

debug_model()
