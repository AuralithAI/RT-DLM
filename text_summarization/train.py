import jax
import os, sys
import jax.numpy as jnp
import haiku as hk
import optax
import pickle
from datasets import load_dataset
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_utils import DataProcessor
from text_summarization.text_summary_module import TextSummarizationModel

# Training Configuration
class TrainConfig:
    vocab_size = 4000
    d_model = 64
    num_heads = 2
    num_layers = 2
    max_seq_length = 64
    batch_size = 1
    learning_rate = 3e-4
    num_epochs = 3

# Load dataset and build vocabulary
processor = DataProcessor()
data = load_dataset("cnn_dailymail", "3.0.0", split="train") 
text_samples = [sample["article"] for sample in data][:10000]
processor.build_vocab(text_samples)

# Define model
def forward_fn(inputs):
    model = TextSummarizationModel(
        TrainConfig.vocab_size, TrainConfig.d_model,
        TrainConfig.num_heads, TrainConfig.num_layers, TrainConfig.max_seq_length
    )
    return model(inputs)

model = hk.transform_with_state(forward_fn)
optimizer = optax.adamw(TrainConfig.learning_rate)

# Initialize model parameters
rng = jax.random.PRNGKey(42)
dummy_inputs = jnp.ones((1, TrainConfig.max_seq_length), dtype=jnp.int32)
params, state = model.init(rng, dummy_inputs)
opt_state = optimizer.init(params)

loss_history = []

# Training loop
for epoch in range(TrainConfig.num_epochs):
    epoch_loss = 0
    num_samples = 0

    for sample in data:
        text = sample["article"]
        summary = sample["highlights"]

        input_tokens = processor.convert_text_to_tokens(text)
        summary_tokens = processor.convert_text_to_tokens(summary)

        input_padded = processor.pad_sequence(input_tokens, TrainConfig.max_seq_length)
        summary_padded = processor.pad_sequence(summary_tokens, TrainConfig.max_seq_length)

        inputs = jnp.array([input_padded])
        targets = jnp.array([summary_padded])

        rng, step_rng = jax.random.split(rng)
        loss, params, state, opt_state = jax.jit(lambda p, s, o, r, i, t: (
            jnp.mean(jnp.sum(jax.nn.one_hot(t, TrainConfig.vocab_size) * 
            jax.nn.log_softmax(model.apply(p, s, r, i)[0], axis=-1), axis=-1)), p, s, o
        ))(params, state, opt_state, step_rng, inputs, targets)

        epoch_loss += loss
        num_samples += 1

        print(f"[Epoch {epoch+1}] Loss: {loss:.4f}")

    avg_loss = epoch_loss / num_samples
    loss_history.append(avg_loss)

with open("trained_model.pkl", "wb") as f:
    pickle.dump((params, state), f)

print("Model training complete! Saved as 'trained_model.pkl'.")

plt.figure(figsize=(8, 5))
plt.plot(range(1, TrainConfig.num_epochs + 1), loss_history, marker="o", linestyle="-", color="b", label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()