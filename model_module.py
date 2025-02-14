import haiku as hk
import jax
import jax.numpy as jnp
from model import TransformerBlock, EmbeddingLayer, MixtureOfExperts
from train_config import TrainConfig

class RTDLMModel(hk.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = EmbeddingLayer(config.vocab_size, config.d_model, config.max_seq_length)
        self.transformer_blocks = [TransformerBlock(config.d_model, config.num_heads) for _ in range(config.num_layers)]
        self.moe_layer = MixtureOfExperts(config.d_model, config.moe_experts, config.moe_top_k, dropout_rate=0.1, temperature=config.temperature)
        self.final_layer = hk.Linear(config.vocab_size, w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"))

    def __call__(self, inputs, rng):
        print("Inside RTDLM Call...")
        print(f"[DEBUG] Model Inputs: {inputs}")
        print(f"[DEBUG] Model Inputs Shape: {inputs.shape}")
        print(f"[DEBUG] Rng : {rng}")
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)

        if inputs.shape[1] != TrainConfig().max_seq_length:
            pad_width = TrainConfig().max_seq_length - inputs.shape[1]
            inputs = jnp.clip(jnp.pad(inputs, ((0, 0), (0, pad_width)), constant_values=0), 0, TrainConfig().vocab_size - 1)
        
        print(f"[After Clipping] Model Inputs Shape: {inputs.shape}")
        print(f"[After Clipping] Model Inputs : {inputs}")

        x = self.embed(inputs, seq_length=inputs.shape[1])
        print(f"[EMBBED] Embedding Layer Output: {x}")

        rng, *subkeys = jax.random.split(rng, num=len(self.transformer_blocks) + 2)
        print(f"[RANDOM SPLIT] Subkeys: {subkeys}")

        for block, subkey in zip(self.transformer_blocks, subkeys[:-1]):
            x = block(x)
        
        print(f"[TRANSFORMER] Transformer Blocks Output: {x}")

        x, load_balancing_loss = self.moe_layer(x, rng=subkeys[-1], is_training=True)
        print(f"[MOE] MoE Layer Output: {x}")
        print(f"[MOE] Load Balancing Loss: {load_balancing_loss}")

        # Ensure `load_balancing_loss` is a valid tensor (handles cases where it might be a float or dictionary)
        if isinstance(load_balancing_loss, float):
            load_balancing_loss = jnp.array(load_balancing_loss)
        elif isinstance(load_balancing_loss, dict):  # Some cases return a dictionary
            load_balancing_loss = jnp.array(0.0)  # Default value to avoid errors

        print(f"[MOE] Load Balancing Loss (After Check): {load_balancing_loss}")
        print(f"[FINAL] Final Layer Input: {x}")

        logits = self.final_layer(x)
        print(f"[LOGITS] Final Layer Output: {logits}")
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        print(f"[LOGITS] Final Layer Output (After Max): {logits}")

        return jnp.asarray(logits), jnp.asarray(load_balancing_loss)

def forward_fn(inputs, rng):
    subkeys = jax.random.split(rng, num=2) if rng.shape == (2,) else jax.random.split(jax.random.PRNGKey(42), num=2)
    model = RTDLMModel(TrainConfig())
    return model(inputs, rng=subkeys[0])

model = hk.without_apply_rng(hk.transform_with_state(forward_fn))