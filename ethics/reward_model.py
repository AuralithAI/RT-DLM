import haiku as hk
import jax
import jax.numpy as jnp
import optax
from typing import Dict, List, Tuple
from data_utils import DataProcessor

class EthicalRewardModel(hk.Module):
    def __init__(self, d_model: int, vocab_size: int, max_seq_length: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.embedding = hk.Embed(vocab_size, d_model)
        self.position_enc = hk.Embed(max_seq_length, d_model)
        self.ffn = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1)  # Output a single score
        ])
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, inputs, outputs):
        inputs = jnp.asarray(inputs, dtype=jnp.int32)
        outputs = jnp.asarray(outputs, dtype=jnp.int32)
        x = self.embedding(inputs) + self.position_enc(jnp.arange(inputs.shape[1]))
        y = self.embedding(outputs) + self.position_enc(jnp.arange(outputs.shape[1]))
        x = jnp.concatenate([x, y], axis=1)
        x = self.norm(x)
        score = self.ffn(x.mean(axis=1))
        return jax.nn.sigmoid(score)


### Only if you want to train-demo the reward model ###
def train_reward_model(config, feedback_dataset: List[Dict], processor: DataProcessor):
    """Train the reward model on feedback dataset."""
    rng = jax.random.PRNGKey(42)

    def forward_fn(inputs, outputs):
        model = EthicalRewardModel(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length * 2
        )
        return model(inputs, outputs)

    model = hk.transform(forward_fn)
    optimizer = optax.adam(learning_rate=1e-4)
    params = model.init(rng, jnp.zeros((1, config.max_seq_length), dtype=jnp.int32),
                        jnp.zeros((1, config.max_seq_length), dtype=jnp.int32))
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, rng, inputs, outputs, targets):
        scores = model.apply(params, rng, inputs, outputs)
        loss = jnp.mean(optax.l2_loss(scores, targets))
        return loss

    @jax.jit
    def update_fn(params, opt_state, rng, inputs, outputs, targets):
        loss, grads = jax.value_and_grad(loss_fn)(params, rng, inputs, outputs, targets)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Prepare dataset
    inputs = []
    outputs = []
    targets = []
    for item in feedback_dataset:
        input_tokens = processor.pad_sequence(processor.tokenize(item["input"]), config.max_seq_length)
        output_tokens = processor.pad_sequence(processor.tokenize(item["output"]), config.max_seq_length)
        inputs.append(input_tokens)
        outputs.append(output_tokens)
        targets.append(item["feedback_score"])
    inputs = jnp.array(inputs, dtype=jnp.int32)
    outputs = jnp.array(outputs, dtype=jnp.int32)
    targets = jnp.array(targets, dtype=jnp.float32)

    # Training loop
    for epoch in range(10):  # Adjust epochs as needed
        rng, sub_rng = jax.random.split(rng)
        loss = 0
        for i in range(0, len(inputs), config.batch_size):
            batch_inputs = inputs[i:i + config.batch_size]
            batch_outputs = outputs[i:i + config.batch_size]
            batch_targets = targets[i:i + config.batch_size]
            params, opt_state, batch_loss = update_fn(params, opt_state, sub_rng,
                                                     batch_inputs, batch_outputs, batch_targets)
            loss += batch_loss
        print(f"Epoch {epoch + 1}, Loss: {loss / (len(inputs) // config.batch_size):.4f}")

    return model, params