import os
import sys
import jax
import optuna
import pickle
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_config import TrainConfig
from train_tms import train_and_evaluate

# Set JAX configurations
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
print("[INFO] JAX device: ", jax.devices())

def objective(trial):
    # Tune parameters
    d_model = trial.suggest_categorical("d_model", [128, 192, 256])
    num_layers = trial.suggest_int("num_layers", 4, 12)
    num_heads = trial.suggest_int("num_heads", 4, 8)
    moe_experts = trial.suggest_categorical("moe_experts", [2, 4, 8])
    moe_top_k = trial.suggest_categorical("moe_top_k", [1, 2])
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)

    # Create model with these params
    config = TrainConfig()
    config.d_model = d_model
    config.num_layers = num_layers
    config.num_heads = num_heads
    config.moe_experts = moe_experts
    config.moe_top_k = moe_top_k
    config.batch_size = batch_size
    config.learning_rate = learning_rate

    # Track losses for plotting
    losses = []

    # Train and evaluate
    t_losses, params = train_and_evaluate(config, losses)  # Modify to return `params`

    # Save model parameters after each trial
    with open(f"tms_params_trial_{trial.number}.pkl", "wb") as f:
        pickle.dump(params, f)
    print(f"[INFO] Model parameters saved for trial {trial.number}")

    # Save loss plot for each trial
    plt.plot(t_losses, label=f"Trial {trial.number}")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.title(f"TMS Training Loss Trial - {trial.number}")
    plt.legend()
    plt.savefig(f"tms_loss_trial_{trial.number}.png")
    print(f"[INFO] Loss plot saved for trial {trial.number}")

    return min(t_losses)

if __name__ == "__main__":
    # Run optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)  # Try 30 different sets of hyperparameters

    # Print best parameters
    print("Best hyperparameters:", study.best_params)
    
    # Load best trial number
    best_trial_num = study.best_trial.number

    # Rename best model and loss plot for easy access
    os.rename(f"tms_params_trial_{best_trial_num}.pkl", "tms_best_params.pkl")
    os.rename(f"tms_loss_trial_{best_trial_num}.png", "tms_best_loss.png")

    print(f"[INFO] Best model parameters saved as tms_best_params.pkl")
    print(f"[INFO] Best loss plot saved as tms_best_loss.png")