# 🫀 AI-Based Heart Disease Detection from Chest X-Ray Images

> A privacy-first, deep-learning-powered diagnostic assistant that analyzes chest X-rays for cardiac abnormalities and provides medical guidance — no data stored, ever.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Why This Architecture?](#why-this-architecture)
- [Technology Stack & Why We Use Each](#technology-stack--why-we-use-each)
- [Functions Explained](#functions-explained)
- [Model & Accuracy](#model--accuracy)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Dataset Instructions](#dataset-instructions)
- [Training the Model](#training-the-model)
- [UI Walkthrough](#ui-walkthrough)
- [Privacy Design](#privacy-design)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

This application accepts a **chest X-ray, CT Scan, Echocardiogram, or ECG image** (JPG/PNG/DICOM) and:

1. Preprocesses the image (resize, normalize, enhance contrast)
2. Ingests optional **Patient Vitals** (Age, BP, Blood Sugar, Body Fat)
3. Runs the data through a **Zero-Shot CLIP ViT-Large-Patch14 model** (along with the custom DenseNet CNN pipelines)
4. Predicts specific anomalies such as Heart Disease, Pulmonary Embolisms, or Myocardial Infarctions
5. Generates **Grad-CAM heatmaps** showing which regions influenced the prediction
6. Provides **dynamically personalized medical recommendations** adapted to both the AI prediction and the patient's uploaded vitals

Everything is processed **in-memory** — no images or patient data are written to disk or any database.

---

## Why This Architecture?

### Why DenseNet121 instead of ResNet50?
DenseNet121 was specifically used in **CheXNet** (Stanford), which achieved radiologist-level performance on the NIH chest X-ray dataset. Its dense connections allow feature reuse and are especially effective for:
- Detecting subtle patterns in grayscale medical images
- Avoiding vanishing gradients with deeper networks
- Requiring fewer parameters than equivalent ResNets

### Why Transfer Learning?
Medical datasets are small relative to natural image datasets. ImageNet-pretrained weights give the model low-level feature detectors (edges, textures, shapes) for free, so fine-tuning on ~10K–100K X-rays achieves far better accuracy than training from scratch.

### Why Grad-CAM?
Grad-CAM (Gradient-weighted Class Activation Mapping) generates a heatmap highlighting **which regions of the X-ray the model focused on**. This is critical for:
- Clinician trust and explainability
- Catching model errors (if it focuses on the wrong region)
- Meeting AI-in-medicine transparency requirements

### Why Streamlit?
Streamlit converts Python scripts into interactive web UIs with minimal boilerplate. For medical tools that need rapid prototyping, it's ideal. The custom CSS in `app.py` makes it look far more polished than default Streamlit.

---

## Technology Stack & Why We Use Each

| Library | Version | Why We Use It |
|---|---|---|
| `torch` / `torchvision` | ≥2.0 | Deep learning framework; dynamic graphs make debugging CNN layers easier than TensorFlow's static graphs |
| `streamlit` | ≥1.32 | Converts Python into a web UI; handles file uploads, state, and layout declaratively |
| `opencv-python` | ≥4.9 | Fast image I/O, CLAHE contrast enhancement, color space conversion, heatmap overlay blending |
| `Pillow` | ≥10.0 | PIL-compatible image object used by Streamlit's `st.image()` and torchvision transforms |
| `numpy` | ≥1.26 | Array operations for preprocessing pipelines and normalization |
| `torchcam` | ≥0.4 | Clean, maintained Grad-CAM implementation for PyTorch; avoids manual hook management |
| `scikit-learn` | ≥1.4 | Metrics (AUC-ROC, F1, confusion matrix) for model evaluation |
| `matplotlib` | ≥3.8 | Grad-CAM heatmap visualization and training curve plots |
| `pydicom` | ≥2.4 | Reads DICOM files (real hospital X-ray format) by extracting pixel arrays |
| `albumentations` | ≥1.4 | Fast, medical-grade augmentation (better than torchvision transforms for training) |
| `tqdm` | ≥4.66 | Progress bars during training loops |

---

## Functions Explained

### `preprocessing/image_processing.py`

| Function | What It Does | Why It's Needed |
|---|---|---|
| `load_image(path)` | Loads JPG/PNG/DICOM → numpy array | DICOM is the real hospital format; we need unified loading |
| `preprocess_for_model(img)` | Resize→224×224, CLAHE, normalize | DenseNet expects 224×224; CLAHE improves low-contrast X-rays |
| `apply_clahe(img)` | Contrast Limited Adaptive Histogram Equalization | Enhances local contrast in X-rays without over-brightening |
| `normalize_image(img)` | (pixel - mean) / std with ImageNet stats | Required for pretrained models expecting normalized inputs |
| `denormalize_for_display(tensor)` | Reverses normalization for visualization | So we can overlay Grad-CAM on the original image, not the normalized one |

### `prediction/predictor.py`

| Function | What It Does | Why It's Needed |
|---|---|---|
| `load_model(path)` | Loads `.pth` weights into DenseNet121 | Separating model loading from inference allows caching |
| `predict(image_tensor)` | Forward pass → probability | Core inference function; returns raw logits + softmax |
| `get_gradcam(image_tensor, model)` | Generates heatmap via gradient backprop | Explainability; shows what triggered the prediction |
| `overlay_heatmap(img, cam)` | Blends colorized CAM onto original X-ray | Visual output for doctors to review |

### `recommendation/medical_advice.py`

| Function | What It Does | Why It's Needed |
|---|---|---|
| `get_advice(condition, confidence)` | Returns structured advice dict | Rule-based engine mapping conditions to clinical guidelines |
| `get_risk_level(confidence)` | LOW / MODERATE / HIGH / CRITICAL | Confidence thresholds map to actionable urgency levels |
| `format_report(prediction)` | Builds complete output dict | Aggregates all outputs into a single report structure |

### `model/train.py`

| Function | What It Does | Why It's Needed |
|---|---|---|
| `build_model(num_classes)` | Creates DenseNet121 with custom head | Replaces ImageNet's 1000-class head with our binary/multi-class head |
| `train_epoch(model, loader, optimizer)` | One full training pass | Core training loop with gradient accumulation |
| `evaluate(model, loader)` | AUC, F1, accuracy on validation set | Monitoring overfitting and convergence |
| `focal_loss(pred, target)` | Class-imbalance-aware loss | X-ray datasets are highly imbalanced (far more normal than diseased) |

---

## Model & Accuracy

### Architecture

```
Input: 224×224×3 chest X-ray
    │
DenseNet121 Backbone (pretrained ImageNet)
    │
Global Average Pooling
    │
Dropout(0.5)
    │
FC(1024 → 256) + ReLU + BatchNorm
    │
Dropout(0.3)
    │
FC(256 → num_classes)
    │
Sigmoid (binary) or Softmax (multi-class)
```

### Expected Accuracy

| Metric | Expected Value |
|---|---|
| AUC-ROC (Cardiomegaly) | 0.87 – 0.92 |
| F1 Score | 0.80 – 0.86 |
| Accuracy | 82% – 89% |

These match or exceed published results from the CheXNet paper on the NIH dataset.

### Accuracy Boosting Techniques Used

1. **Focal Loss** — handles class imbalance (normal >> abnormal)
2. **CLAHE preprocessing** — improves image quality before inference
3. **Test-Time Augmentation (TTA)** — averages predictions across multiple augmented versions of the same image
4. **Label Smoothing** — prevents overconfident wrong predictions
5. **Cosine LR Annealing** — finds better minima than step decay
6. **Albumentations augmentation** — realistic medical augmentations during training

---

## Project Structure

```
heart-xray-ai/
│
├── README.md                    ← You are here
├── requirements.txt             ← All dependencies with pinned versions
├── app.py                       ← Main Streamlit UI (run this)
│
├── model/
│   ├── train.py                 ← Training script
│   ├── evaluate.py              ← Evaluation + metrics
│   └── heart_xray_model.pth    ← Trained weights (download separately)
│
├── preprocessing/
│   └── image_processing.py     ← All image preprocessing functions
│
├── prediction/
│   └── predictor.py            ← Model inference + Grad-CAM
│
├── recommendation/
│   └── medical_advice.py       ← Rule-based medical guidance engine
│
├── utils/
│   └── helpers.py              ← Shared utilities (logging, file handling)
│
├── assets/
│   └── demo_xray.jpg           ← Sample X-ray for testing (normal chest)
│
└── datasets/
    └── README_datasets.md      ← Instructions to download NIH/CheXpert datasets
```

---

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip
- 4GB RAM minimum (8GB recommended)
- GPU optional but recommended for training (not required for inference)

### Step 1: Clone or Extract the Project

```bash
cd heart-xray-ai
```

### Step 2: Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all libraries listed in `requirements.txt`.

---

## How to Run

### Option A: Run the Application Locally

1. Download the clinical dataset if you wish to train your own custom weights (see `datasets/README_datasets.md`)
2. Run the application via Streamlit:
   ```bash
   streamlit run app.py
   ```
3. Open your browser at: **http://localhost:8501**
4. Upload any medical image relevant to the supported modalities (X-Ray, CT, ECG, Echo) to begin the analysis.

### Option C: Docker (No Python Setup Required)

```bash
docker build -t heart-xray-ai .
docker run -p 8501:8501 heart-xray-ai
```

---

## Dataset Instructions

See `datasets/README_datasets.md` for full instructions.

**Quick summary:**
- **NIH ChestX-ray14**: 112,120 frontal X-rays, 14 disease labels including Cardiomegaly. Free download from NIH.
- **CheXpert**: 224,316 chest X-rays from Stanford. Free for research.

---

## Training the Model

```bash
python model/train.py \
  --data_dir ./datasets/images \
  --labels_file ./datasets/Data_Entry_2017.csv \
  --epochs 30 \
  --batch_size 32 \
  --lr 0.0001 \
  --output_path ./model/heart_xray_model.pth
```

Training takes ~2–4 hours on a GPU, ~12–24 hours on CPU.

---

## UI Walkthrough

1. **Upload Tab** — Drag and drop your medical image (X-Ray, CT, Echo, or ECG)
2. **Patient Vitals** — Fill in optional patient metadata to heavily personalize the medical report algorithms
3. **Analysis** — Click "Analyze Health Record" button
4. **Results Panel** — Shows prediction, confidence, risk level based on image + vitals
5. **Heatmap View** — Grad-CAM overlay showing where the model localized the anomaly
6. **Report Tab** — Full medical recommendations, dietary advice, lifestyle changes, and urgency protocols

---

## Privacy Design

- Images are loaded into RAM only
- No image is written to disk during inference
- No database or logging of patient data
- Session state is cleared on browser refresh
- All processing happens locally on your machine

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: torchcam` | Run `pip install torchcam` |
| `CUDA out of memory` | Set `--batch_size 8` or run on CPU |
| `Image too dark / bad predictions` | Ensure it's a frontal chest X-ray (PA or AP view) |
| Streamlit not opening | Try `streamlit run app.py --server.port 8502` |
| DICOM file not loading | Install `pydicom`: `pip install pydicom` |

---

## Disclaimer

> This tool is for **educational and research purposes only**. It is **not a substitute for professional medical diagnosis**. Always consult a qualified radiologist or cardiologist for medical decisions.

---

## License

MIT License — free for academic and research use.
