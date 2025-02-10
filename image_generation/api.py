from fastapi import FastAPI
from pydantic import BaseModel
import jax.numpy as jnp
import jax
import os
import time
import uuid
import haiku as hk
import pickle
import numpy as np
import matplotlib.pyplot as plt
from image_generation.model_module import model
from image_generation.config import ImageGenConfig

app = FastAPI()
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "..", "trained_image_model.pkl")
config = ImageGenConfig()

# Load trained model from the .pkl file
with open(MODEL_PATH, "rb") as f:
    params, state = pickle.load(f)

class ImageInput(BaseModel):
    seed: int = 42

@app.post("/generate_image")
def generate_image(input_data: ImageInput):
    global params, state
    rng = jax.random.PRNGKey(input_data.seed)
    z = jax.random.normal(rng, (1, config.latent_dim))
    generated_image, _ = model.apply(params, state, rng, z, config)

    img_data = np.array(generated_image.squeeze())
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())

    unique_id = uuid.uuid4().hex[:8]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    img_filename = f"generated_image_{timestamp}_{unique_id}.png"

    img_path = os.path.join("generated_images", img_filename)
    os.makedirs("generated_images", exist_ok=True) 
    plt.imsave(img_path, img_data, cmap="gray")
    return {"image_path": img_path}