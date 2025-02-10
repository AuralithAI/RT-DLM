import sys
import os
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from image_generation.model_module import ImageGenerator
from image_generation.config import ImageGenConfig
import matplotlib.pyplot as plt
import pickle

jax.config.update("jax_platform_name", "gpu")

config = ImageGenConfig()
optimizer = optax.adam(1e-4)

def preprocess_image(image):
    """Resize and normalize images."""
    image = tf.image.resize(image, (config.image_size, config.image_size))  
    image = tf.cast(image, tf.float32) / 255.0 
    image = image.numpy() if isinstance(image, tf.Tensor) else image 
    return jnp.array(image)

def load_dataset():
    dataset, info = tfds.load("mnist", split="train", as_supervised=True, with_info=True)
    
    def tf_to_numpy(img, lbl):
        img = tf.image.resize(img, (config.image_size, config.image_size))  
        img = tf.cast(img, tf.float32) / 255.0  
        return img, lbl 

    dataset = dataset.map(lambda img, lbl: tf_to_numpy(img, lbl)).batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset, info

def forward_fn(z):
    model = ImageGenerator(config)
    return model(z)

model = hk.transform_with_state(forward_fn)

def train():
    """Train the image generator model and save it as a .pkl file."""
    rng = jax.random.PRNGKey(42)

    dummy_z = jax.random.normal(rng, (1, config.latent_dim))
    params, state = model.init(rng, dummy_z)

    dataset, info = load_dataset()
    print(f"Dataset: {info.name} with {info.splits['train'].num_examples} images.")

    loss_history = []

    for epoch in range(10):
        epoch_loss = 0
        for batch in dataset:
            images, _ = batch 

            images = jnp.array(images)  
            images = jnp.expand_dims(images, axis=-1) if images.ndim == 3 else images 

            images = jax.image.resize(images, (images.shape[0], 64, 64, 3), method="bilinear")

            z = jax.random.normal(rng, (images.shape[0], config.latent_dim))

            generated_imgs, _ = model.apply(params, state, rng, z)

            print(f"[DEBUG] generated_imgs.shape: {generated_imgs.shape}, images.shape: {images.shape}")

            generated_imgs = jax.image.resize(generated_imgs, images.shape, method="bilinear")  

            loss = jnp.mean((generated_imgs - images) ** 2)

            grads = jax.grad(lambda p: jnp.mean((model.apply(p, state, rng, z)[0] - images) ** 2))(params)
            updates, opt_state = optimizer.update(grads, optimizer.init(params))
            params = optax.apply_updates(params, updates)

            epoch_loss += loss

        loss_history.append(epoch_loss / len(dataset))
        print(f"Epoch {epoch+1}, Loss: {loss_history[-1]:.4f}")

    with open("trained_image_model.pkl", "wb") as f:
        pickle.dump((params, state), f)

    print("Model training complete! Saved as 'trained_image_model.pkl'.")

    plt.plot(range(1, 11), loss_history, marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.show()

if __name__ == "__main__":
    train()
