import gradio as gr
from classification.model import predict as classify_image
from captioning.caption import generate_caption

def classify_and_caption(image):
    """
    Returns (top-3 classifications, generated caption)
    """
    classes = classify_image(image)
    caption = generate_caption(image)
    return classes, caption

iface = gr.Interface(
    fn=classify_and_caption,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=3), gr.Textbox(label="Caption")],
    title="Image Classification + Captioning",
    description="Upload an image to classify (top 3) and generate a caption."
)

if __name__ == "__main__":
    iface.launch()