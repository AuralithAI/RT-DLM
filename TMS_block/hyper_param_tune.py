import os
import sys
import optuna
import pickle
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_config import TrainConfig
from train_tms import train_and_evaluate  # Import your training function

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
    val_loss, params = train_and_evaluate(config, losses)  # Modify to return `params`

    # Save model parameters after each trial
    with open(f"tms_params_trial_{trial.number}.pkl", "wb") as f:
        pickle.dump(params, f)
    print(f"[INFO] Model parameters saved for trial {trial.number}")

    # Save loss plot for each trial
    plt.plot(losses, label=f"Trial {trial.number}")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.title("TMS Training Loss")
    plt.legend()
    plt.savefig(f"tms_loss_trial_{trial.number}.png")
    print(f"[INFO] Loss plot saved for trial {trial.number}")

    return val_loss

if __name__ == "__main__":
    # Run optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)  # Try 30 different sets of hyperparameters

    # Print best parameters
    print("Best hyperparameters:", study.best_params)