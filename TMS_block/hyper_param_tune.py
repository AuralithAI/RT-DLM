import os
import sys
import jax
import gc
import optuna
import pickle
import logging
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_config import TrainConfig
from train_tms import train_and_evaluate
from jax.extend import backend

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set JAX configurations
def set_jax_config(config):
    jax.config.update("jax_platform_name", "gpu")
    jax.config.update("jax_enable_x64", False)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    #os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
    os.environ["XLA_FLAGS"] = (
        f"--xla_gpu_force_compilation_parallelism={config.xla_gpu_parallelism}"
        "--xla_gpu_enable_triton_gemm=true"  
        "--xla_gpu_memory_efficient=true"     
    )
    print("[INFO] JAX device: ", jax.devices())

def clear_gpu_memory():
    """Clear GPU memory after each trial."""
    jax.clear_caches()  
    for device in jax.devices():
        try:
            jax.device_put(jnp.zeros((1,)), device=device) 
            jax.device_put(None, device=device) 
        except Exception as e:
            logger.warning(f"[WARNING] Failed to clear device memory: {e}")
    gc.collect()
    logger.info("[INFO] GPU memory cleared")

def objective(trial):

    logger.info(f"[INFO] VRAM usage before trial {trial.number}: {jax.local_device_count()} devices")

    # Tune Model Architecture Parameters
    d_model = trial.suggest_categorical("d_model", [256, 384, 512])
    # Static list of possible num_heads, validated against d_model
    num_heads = trial.suggest_categorical("num_heads", [4, 6, 8, 12])
    # Ensure compatibility: d_model % num_heads == 0
    while d_model % num_heads != 0:
        num_heads = trial.suggest_categorical("num_heads", [4, 6, 8, 12])
    
    num_layers = trial.suggest_int("num_layers", 6, 12)
    moe_experts = trial.suggest_categorical("moe_experts", [4, 8])
    moe_top_k = trial.suggest_categorical("moe_top_k", [2, 3])

    # Tune Training Hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 5e-3, log=True) 
    inner_learning_rate = trial.suggest_float("inner_learning_rate", 5e-4, 2e-3, log=True)
    warmup_steps = trial.suggest_int("warmup_steps", 1000, 10000, step=1000)
    decay_steps = trial.suggest_int("decay_steps", 50000, 300000, step=50000)

    # Tune Memory Bank Parameters
    memory_size = trial.suggest_categorical("memory_size", [1000, 5000, 10000, 20000])
    retrieval_k = trial.suggest_categorical("retrieval_k", [1, 3, 5, 7])
    stm_buffer_size = trial.suggest_categorical("stm_buffer_size", [8, 16, 32, 64, 128])
    mtm_buffer_size = trial.suggest_categorical("mtm_buffer_size", [500, 1000, 2000, 4000])
    retention_steps = trial.suggest_int("retention_steps", 50, 200, step=50)
    ltm_weight = trial.suggest_float("ltm_weight", 0.0, 1.0)
    stm_weight = trial.suggest_float("stm_weight", 0.0, 1.0)
    mtm_weight = trial.suggest_float("mtm_weight", 0.0, 1.0)

    # Create model with these params
    trial_params = {
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "moe_experts": moe_experts,
        "moe_top_k": moe_top_k,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "inner_learning_rate": inner_learning_rate,
        "warmup_steps": warmup_steps,
        "decay_steps": decay_steps,
        "memory_size": memory_size,
        "retrieval_k": retrieval_k,
        "stm_buffer_size": stm_buffer_size,
        "mtm_buffer_size": mtm_buffer_size,
        "retention_steps": retention_steps,
        "ltm_weight": ltm_weight,
        "stm_weight": stm_weight,
        "mtm_weight": mtm_weight,
        "vocab_size": 8000,  # Default from TrainConfig
        "max_seq_length": 64,  # Default from TrainConfig
        "task_size": 15,  # Default from TrainConfig
        "num_inner_steps": 10,  # Default from TrainConfig
        "num_epochs": 3,  # Default from TrainConfig
        "eval_interval": 25,  # Default from TrainConfig
        "temperature": 1.2,  # Default from TrainConfig
        "label_smoothing": 0.1,  # Default from TrainConfig
        "warmup_steps": warmup_steps,  # Overridden by trial
        "decay_steps": decay_steps,  # Overridden by trial
        "init_lr": 2e-6,  # Default from TrainConfig
        "end_lr": 2e-6,  # Default from TrainConfig
        "weight_decay": 1e-3,  # Default from TrainConfig
        "clip_norm": 0.5,  # Default from TrainConfig
        "max_sentence_length": 5192,  # Default from TrainConfig
        "input_sentence_size": 500000,  # Default from TrainConfig
        "character_coverage": 0.9999,  # Default from TrainConfig
        "num_threads": 16,  # Default from TrainConfig
        "xla_gpu_parallelism": 10,  # Default from TrainConfig
        "EPSILON": 1e-8,  # Default from TrainConfig
        "prune_threshold": 0.01  # Default from TrainConfig
    }
    config = TrainConfig(**trial_params)

    # Apply JAX/XLA config
    set_jax_config(config)

    # Track losses and similarity scores
    losses = []
    similarity_scores = []
    thought_logs = []

    logger.info(f"[Trial {trial.number}] d_model: {d_model}, num_layers: {num_layers}")

    # Train and evaluate
    t_losses, params, t_similarity_scores, state, ltm, stm, mtm, t_thought_logs = train_and_evaluate(config, losses, similarity_scores, thought_logs)

    # Save model parameters, state, and all memory banks
    trial_number = trial.number
    with open(f"TMS_block/tms_params_trial_{trial_number}.pkl", "wb") as f:
        pickle.dump(params, f)
    with open(f"TMS_block/tms_state_trial_{trial_number}.pkl", "wb") as f:
        pickle.dump(state, f)
    with open(f"TMS_block/ltm_bank_trial_{trial_number}.pkl", "wb") as f:
        pickle.dump(ltm, f)
    with open(f"TMS_block/stm_bank_trial_{trial_number}.pkl", "wb") as f:
        pickle.dump(stm, f)
    with open(f"TMS_block/mtm_bank_trial_{trial_number}.pkl", "wb") as f:
        pickle.dump(mtm, f)
    with open(f"TMS_block/thought_log_trial_{trial_number}.pkl", "wb") as f:
        pickle.dump(t_thought_logs, f)
    logger.info(f"[INFO] Saved params, state, all memory banks and thought_log for trial {trial_number}")

    # Save loss and similarity plots
    plt.plot(t_losses, label=f"Trial {trial_number}")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.title(f"TMS Training Loss Trial - {trial_number}")
    plt.legend()
    plt.savefig(f"TMS_block/tms_loss_trial_{trial_number}.png")
    plt.close()
    logger.info(f"[INFO] Loss plot saved for trial {trial_number}")

    plt.plot(t_similarity_scores, label=f"Memory Similarity {trial_number}")
    plt.xlabel("Training Steps")
    plt.ylabel("Memory Retrieval Score")
    plt.grid(True)
    plt.title(f"Memory Retrieval Similarity Over Time Trial - {trial_number}")
    plt.legend()
    plt.savefig(f"TMS_block/memory_retrieval_similarity_{trial_number}.png")
    plt.close()
    logger.info(f"[INFO] Memory retrieval similarity plot saved as memory_retrieval_similarity_{trial_number}.png")

    # Clear GPU memory after trial
    clear_gpu_memory()

    return min(t_losses)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)

    logger.info("Best hyperparameters:", study.best_params)
    best_trial_num = study.best_trial.number

    os.rename(f"TMS_block/tms_params_trial_{best_trial_num}.pkl", "TMS_block/tms_best_params.pkl")
    os.rename(f"TMS_block/tms_state_trial_{best_trial_num}.pkl", "TMS_block/tms_best_state.pkl")
    os.rename(f"TMS_block/ltm_bank_trial_{best_trial_num}.pkl", "TMS_block/ltm_bank.pkl")
    os.rename(f"TMS_block/stm_bank_trial_{best_trial_num}.pkl", "TMS_block/stm_bank.pkl")
    os.rename(f"TMS_block/mtm_bank_trial_{best_trial_num}.pkl", "TMS_block/mtm_bank.pkl")
    os.rename(f"TMS_block/tms_loss_trial_{best_trial_num}.png", "TMS_block/tms_best_loss.png")

    logger.info(f"[INFO] Best model parameters saved as TMS_block/tms_best_params.pkl")
    logger.info(f"[INFO] Best state saved as TMS_block/tms_best_state.pkl")
    logger.info(f"[INFO] Best LTM bank saved as TMS_block/ltm_bank.pkl")
    logger.info(f"[INFO] Best STM bank saved as TMS_block/stm_bank.pkl")
    logger.info(f"[INFO] Best MTM bank saved as TMS_block/mtm_bank.pkl")
    logger.info(f"[INFO] Best loss plot saved as TMS_block/tms_best_loss.png")