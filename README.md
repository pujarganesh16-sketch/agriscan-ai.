
# 🌿 AgriScan AI
### AI-Powered Plant Health Analysis and Crop Disease Screening

AgriScan AI is a machine-learning project designed to make plant-health screening more accessible through image-based analysis. It combines a trained deep-learning model with an interactive web interface to help users examine crop leaf images and explore potential disease classifications.

The project also incorporates image-quality diagnostics, prediction-confidence analysis, historical scan analytics, and an advanced laboratory interface for examining model outputs.

> **Important:** AgriScan AI is a screening and educational support tool, not a substitute for professional agricultural diagnosis. Model predictions must be independently verified.

## 🎯 Project Objectives

- Apply artificial intelligence to a practical agricultural problem.
- Build an interactive application around a trained image-classification model.
- Present predictions and model scores in an understandable format.
- Examine uncertainty instead of treating every prediction as certain.
- Provide diagnostic information about uploaded image quality.
- Explore scan-history analytics and model-evaluation methods.
- Deploy the application through a reproducible, version-controlled workflow.

## ✨ Features

### 🔬 AI-Based Leaf Image Analysis
- Upload a crop leaf image for analysis.
- Process images using the configured deep-learning model.
- Present the predicted class and available model scores.
- Display results through an interactive web interface.

### 📊 Prediction Confidence and Uncertainty
- Summarize the highest model score.
- Calculate the margin between the leading predictions.
- Analyze score distributions using entropy-based statistics.
- Communicate uncertainty without presenting model scores as guaranteed disease probabilities.

### 🖼️ Image Quality Diagnostics
- Inspect image resolution, brightness, and contrast.
- Calculate sharpness and color-related measurements.
- Estimate image entropy and green-pixel fraction.
- Display quality notes to help users assess whether an image is suitable for analysis.

### 📈 Advanced Analysis Laboratory
- Explore scan-history summaries and prediction distributions.
- Group historical scans by time period.
- Examine confidence-score statistics.
- Support statistical model-evaluation workflows when labeled validation data is available.

### 🌱 Agricultural Decision Support
- Help users investigate possible crop-health issues.
- Encourage independent confirmation of predictions.
- Emphasize responsible interpretation of automated results.

## 🧠 How It Works

1. **Image input:** The user uploads a crop leaf photograph.
2. **Image processing:** The application prepares the image for model inference.
3. **Model inference:** The configured neural network produces class scores.
4. **Result analysis:** The application summarizes predictions and uncertainty.
5. **Diagnostics:** Image-quality measurements help identify potential input limitations.
6. **Presentation:** Results are displayed in the web interface for further investigation.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and data processing |
| TensorFlow / Keras | Deep-learning model inference |
| NumPy | Numerical and statistical calculations |
| Pandas | Tabular data and scan-history analytics |
| Pillow | Image handling and image-quality analysis |
| Gradio | Interactive web application |
| Git and GitHub | Version control and source-code management |
| Render | Application hosting and deployment |

## 📁 Repository Structure

```text
agriscan-ai/
├── app.py
├── requirements.txt
├── .python-version
├── agriscan_model.keras
├── class_labels.json
└── README.md
```

The model, label file, and configuration files shown above should match the actual files committed to the repository.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/pujarganesh16-sketch/agriscan-ai.git
cd agriscan-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open the local address printed by Gradio in your terminal.

Use a Python version compatible with the project's pinned dependencies.

## ☁️ Deployment

The project is configured for deployment on Render using the GitHub repository.

General deployment workflow:

1. Connect the GitHub repository to Render.
2. Configure the Python runtime and required dependencies.
3. Set the start command to `python app.py`.
4. Confirm that the model and label files are available at runtime.
5. Review deployment logs and resolve startup errors.
6. Verify that the application starts successfully and responds to user requests.

**Deployment status:** Successful deployment and live application behavior must be verified before they are reported as complete.

## 🔍 Model Evaluation and Limitations

Prediction scores describe the model's output distribution. They do not automatically establish real-world diagnostic accuracy or the probability that a plant has a particular disease.

Reliable evaluation requires:
- A representative, independently labeled validation dataset.
- Appropriate train/validation/test separation.
- Per-class precision, recall, and F1-score.
- A confusion matrix and relevant aggregate metrics.
- Evaluation across different crops, lighting conditions, backgrounds, and field environments.
- Calibration analysis when probability estimates are intended to be interpreted probabilistically.

Image-quality heuristics and historical scan summaries are supplementary diagnostics, not proof of model accuracy.

## 🛣️ Future Improvements

- Validate performance on independently collected field images.
- Add more crop and disease categories using suitable training data.
- Improve robustness to lighting, blur, and complex backgrounds.
- Add automated testing and deployment checks.
- Expand explainability and model-calibration analysis.
- Improve accessibility and mobile usability.
- Explore carefully validated, location-aware agricultural guidance.

## 🎓 Educational and Research Value

AgriScan AI demonstrates practical work at the intersection of:

- Artificial intelligence and deep learning
- Computer vision and image classification
- Python application development
- Statistical analysis and uncertainty measurement
- Responsible AI communication
- Software deployment and version control

The project is intended to evolve through testing, validation, documentation, and further experimentation.

## ⚠️ Disclaimer

AgriScan AI provides informational screening results only. Predictions may be incorrect, particularly for unfamiliar crops, mixed infections, poor-quality photographs, or field conditions that differ from the model's training data.

Do not make pesticide, treatment, or crop-management decisions solely from an AI prediction. Consult a qualified agricultural expert and follow locally applicable guidance.

## 👨‍💻 Author

**Project:** AgriScan AI  
**Developer:** Project owner and contributor  
**Repository:** https://github.com/pujarganesh16-sketch/agriscan-ai

---

*Built to explore how AI can support more accessible and responsible crop-health screening.*
