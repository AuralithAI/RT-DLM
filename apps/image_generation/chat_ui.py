import gradio as gr
import requests
import numpy as np
import matplotlib.pyplot as plt

def generate_image(seed):
    response = requests.post("http://127.0.0.1:8000/generate_image", json={"seed": seed})
    img_data = np.array(response.json()["image"]).squeeze()
    
    plt.imshow(img_data, cmap="gray")
    plt.axis("off")
    plt.show()

iface = gr.Interface(
    fn=generate_image,
    inputs="number",
    outputs="text",
    title="Image Generation",
    description="Enter a random seed to generate an image."
)

iface.launch()