# AgriScan AI

A Gradio-based crop/leaf health screening research prototype.

## Run locally
1. Install dependencies: `pip install -r requirements.txt`
2. Provide the trained model file `agriscan_model.keras` and `class_labels.json`.
3. Run: `python app.py`

The model and class labels are not included in this source package. The app may open without them, but predictions will not work until both are provided.

## Render deployment
- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`
- Environment variable: `GRADIO_SHARE=false`

This is a research prototype, not a substitute for expert crop diagnosis or pesticide guidance.
