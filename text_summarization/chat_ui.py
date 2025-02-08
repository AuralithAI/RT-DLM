import gradio as gr
import requests

def chatbot_interface(text):
    response = requests.post("http://127.0.0.1:8000/summarize", json={"text": text})
    return response.json()["summary"]

iface = gr.Interface(
    fn=chatbot_interface,
    inputs="text",
    outputs="text",
    title="Text Summarization Chatbot",
    description="Enter a paragraph, and the chatbot will summarize it."
)

iface.launch()
