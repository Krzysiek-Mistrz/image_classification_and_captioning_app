from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image: Image.Image) -> str:
    """
    Given a PIL Image, returns a generated caption.
    """
    inputs = _processor(images=image, return_tensors="pt")
    outputs = _model.generate(**inputs)
    return _processor.decode(outputs[0], skip_special_tokens=True)