import torch
import requests
from torchvision import transforms
from PIL import Image

def _load_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.eval()
    return model

def _load_labels():
    response = requests.get("https://git.io/JJkYN")
    return response.text.split("\n")

_model = _load_model()
_labels = _load_labels()

def predict(image: Image.Image) -> dict:
    """
    Given a PIL Image, returns a dict of ImageNet label → confidence.
    """
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(_model(tensor)[0], dim=0)
    return { _labels[i]: float(probs[i]) for i in range(len(_labels)) }