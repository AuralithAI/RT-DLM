import jax
import os, sys
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import pickle
from datasets import load_dataset
import matplotlib
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_utils import DataProcessor
from text_summarization.text_summary_module import TextSummarizationModel
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pytz")

jax.config.update("jax_platform_name", "gpu")

if not sys.stdout.isatty():
    print("Running in headless mode: Switching Matplotlib backend to 'Agg'")
    matplotlib.use("Agg")

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

processor = DataProcessor()
data = load_dataset("cnn_dailymail", "3.0.0", split="train[:5%]") 
text_samples = [sample["article"] for sample in data][:10000]
processor.build_vocab(text_samples)

def forward_fn(inputs):
    model = TextSummarizationModel(
        TrainConfig.vocab_size, TrainConfig.d_model,
        TrainConfig.num_heads, TrainConfig.num_layers, TrainConfig.max_seq_length
    )
    return model(inputs)

model = hk.transform_with_state(forward_fn)
optimizer = optax.adamw(TrainConfig.learning_rate)

rng = jax.random.PRNGKey(42)
dummy_inputs = jnp.ones((1, TrainConfig.max_seq_length), dtype=jnp.int32)
params, state = model.init(rng, dummy_inputs)
opt_state = optimizer.init(params)

loss_history = []
sample_loss_history = []

if matplotlib.get_backend() == "TkAgg":
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Real-Time Training Loss")
    ax.grid()
    line, = ax.plot([], [], marker="o", linestyle="-", color="b", label="Loss per Sample")
    plt.legend()
    plt.show(block=False)
else:
    print("Matplotlib is in headless mode: No interactive plots will be shown.")

# Training loop
for epoch in range(TrainConfig.num_epochs):
    epoch_loss = 0
    num_samples = 0
    sample_count = 0

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
        sample_count += 1
        sample_loss_history.append(loss)

        print(f"[Epoch {epoch+1}, Sample {sample_count}] Loss: {loss:.4f}")

        if(len(sample_loss_history) > 1) and matplotlib.get_backend() == "TkAgg":
            x_data = np.arange(1, len(sample_loss_history) + 1)
            y_data = np.array(sample_loss_history)

            line.set_xdata(x_data)
            line.set_ydata(y_data)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.flush_events()
            #plt.pause(0.1)

    avg_loss = epoch_loss / num_samples
    loss_history.append(avg_loss)

with open("trained_model.pkl", "wb") as f:
    pickle.dump((params, state), f)

print("Model training complete! Saved as 'trained_model.pkl'.")

plt.ioff()
plt.figure(figsize=(8, 5))
plt.plot(range(1, TrainConfig.num_epochs + 1), loss_history, marker="o", linestyle="-", color="r", label="Epoch Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid()

if matplotlib.get_backend() == "TkAgg":
    plt.show()
else:
    print("Saving training loss plot as 'training_loss.png' (headless mode).")
    plt.savefig("training_loss.png")