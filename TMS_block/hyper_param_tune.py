import os
import sys
import jax
import optuna
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_config import TrainConfig
from train_tms import train_and_evaluate

# Load configuration globally.
config = TrainConfig()

# Set JAX configurations
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_FLAGS"] = f"--xla_gpu_force_compilation_parallelism={config.xla_gpu_parallelism}"
print("[INFO] JAX device: ", jax.devices())

def objective(trial):
    # Tune parameters
    d_model = trial.suggest_categorical("d_model", [256, 384, 512])
    valid_heads = {256: [4, 8], 384: [4, 6, 8, 12], 512: [4, 8]}
    num_heads = trial.suggest_categorical("num_heads", valid_heads[d_model])
    num_layers = trial.suggest_int("num_layers", 6, 12)
    moe_experts = trial.suggest_categorical("moe_experts", [4, 8, 16])
    moe_top_k = trial.suggest_categorical("moe_top_k", [1, 2, 3])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    memory_size = trial.suggest_categorical("memory_size", [1000, 5000, 10000, 20000])
    retrieval_k = trial.suggest_categorical("retrieval_k", [1, 3, 5, 7])

    # Create model with these params
    config.d_model = d_model
    config.num_layers = num_layers
    config.num_heads = num_heads
    config.moe_experts = moe_experts
    config.moe_top_k = moe_top_k
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.memory_size = memory_size
    config.retrieval_k = retrieval_k


    # Track losses for plotting
    losses = []
    similiarity_scores = []

    # Train and evaluate
    t_losses, params, t_memory_retrieval_scores, state, memory = train_and_evaluate(config, losses, similiarity_scores)

    # Save model parameters, state, and memory after each trial
    trial_number = trial.number
    with open(f"TMS_block/tms_params_trial_{trial_number}.pkl", "wb") as f:
        pickle.dump(params, f)
    with open(f"TMS_block/tms_state_trial_{trial_number}.pkl", "wb") as f:
        pickle.dump(state, f)
    with open(f"TMS_block/memory_bank_trial_{trial_number}.pkl", "wb") as f:
        pickle.dump(memory, f)
    print(f"[INFO] Saved params, state, and memory for trial {trial_number}")

    # Save loss plot for each trial
    plt.plot(t_losses, label=f"Trial {trial_number}")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.title(f"TMS Training Loss Trial - {trial_number}")
    plt.legend()
    plt.savefig(f"TMS_block/tms_loss_trial_{trial_number}.png")
    plt.close()
    print(f"[INFO] Loss plot saved for trial {trial_number}")

    plt.plot(t_memory_retrieval_scores, label=f"Memory Similarity {trial_number}")
    plt.xlabel("Training Steps")
    plt.ylabel("Memory Retrieval Score")
    plt.title(f"Memory Retrieval Similarity Over Time Trial - {trial_number}")
    plt.legend()
    plt.savefig(f"TMS_block/memory_retrieval_similarity_{trial_number}.png")
    plt.close()
    print(f"[INFO] Memory retrieval similarity plot saved as memory_retrieval_similarity_{trial_number}.png")

    return min(t_losses)

if __name__ == "__main__":
    # Run optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    # Print best parameters
    print("Best hyperparameters:", study.best_params)
    
    # Load best trial number
    best_trial_num = study.best_trial.number

    # Rename best model, state, memory, and loss plot for easy access
    os.rename(f"TMS_block/tms_params_trial_{best_trial_num}.pkl", "TMS_block/tms_best_params.pkl")
    os.rename(f"TMS_block/tms_state_trial_{best_trial_num}.pkl", "TMS_block/tms_best_state.pkl")
    os.rename(f"TMS_block/memory_bank_trial_{best_trial_num}.pkl", "TMS_block/memory_bank.pkl")
    os.rename(f"TMS_block/tms_loss_trial_{best_trial_num}.png", "TMS_block/tms_best_loss.png")

    print(f"[INFO] Best model parameters saved as TMS_block/tms_best_params.pkl")
    print(f"[INFO] Best state saved as TMS_block/tms_best_state.pkl")
    print(f"[INFO] Best memory bank saved as TMS_block/memory_bank.pkl")
    print(f"[INFO] Best loss plot saved as TMS_block/tms_best_loss.png")