import jax
import jax.numpy as jnp
from train_config import TrainConfig
from model_module import model

def debug_model():
    config = TrainConfig()
    rng = jax.random.PRNGKey(42)

    # 🚀 Manually create a **valid** test batch
    test_inputs = jnp.array([
        [5, 10, 15, 20, 25] + [0] * (config.max_seq_length - 5),
        [30, 35, 40, 45, 50] + [0] * (config.max_seq_length - 5),
        [100, 200, 300, 400, 500] + [0] * (config.max_seq_length - 5),
        [1000, 2000, 3000, 4000, 5000] + [0] * (config.max_seq_length - 5)
    ], dtype=jnp.int32)

    print(f"[DEBUG] Static Test Input - Before Model")
    print(f"[DEBUG] Min Token ID: {test_inputs.min()}, Max Token ID: {test_inputs.max()}")
    print(f"[DEBUG] Inputs: {test_inputs}")

    # 🚀 Manually run the model
    params, state = model.init(rng, test_inputs, rng)
    predictions, load_balancing_loss = model.apply(params, state, rng, test_inputs)

    print(f"[DEBUG] Static Test Output - After Model")
    print(f"[DEBUG] Predictions: {predictions}")

debug_model()
