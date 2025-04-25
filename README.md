# Image Classification & Captioning

This project lets you upload an image and see:
- Top-3 ImageNet classifications (via ResNet-18)
- A natural-language caption (via Salesforce BLIP)

## Project Structure

```
.
├── README.md
├── requirements.txt
└── src
    ├── app.py
    ├── classification
    │   └── model.py
    └── captioning
        └── caption.py
```

## Installation

1. Clone this repo  
   `git clone <url> && cd <dir>`

2. Create venv & install  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

```bash
python src/app.py
```

This will launch a Gradio UI at `http://localhost:7860`.

Upload an image and you’ll see the top-3 predicted labels plus a generated caption.

## License

GNU GPL V3 @ 2025 Krzychu