"""
prediction/predictor.py
========================
Model loading, inference, Grad-CAM generation, and heatmap overlay.

Why separate from app.py?
  Separation of concerns: app.py handles UI, predictor.py handles ML.
  This makes unit testing, model swapping, and debugging much easier.

Why DenseNet121?
  CheXNet (Rajpurkar et al., Stanford 2017) demonstrated that DenseNet121
  achieves radiologist-level performance on the NIH ChestX-ray14 dataset.
  Its dense connections encourage feature reuse and are especially good at
  detecting subtle structural changes in medical images.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
from PIL import Image

# HuggingFace for Zero-Shot
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    pass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.image_processing import preprocess_for_model, denormalize_for_display

# ─── Disease Labels ──────────────────────────────────────────────────────────
# Binary classification: index 0 = No Finding, index 1 = Heart Disease
BINARY_LABELS = ["No Finding", "Heart Disease"]

# Multi-class cardiac conditions
CONDITION_LABELS = [
    "Cardiomegaly",          # Enlarged heart — most visible on X-ray
    "Congestive Heart Failure",
    "Coronary Artery Disease",
    "Cardiomyopathy",
    "Pulmonary Edema",       # Fluid in lungs due to heart failure
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Model Architecture ───────────────────────────────────────────────────────

def build_model(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    """
    Build DenseNet121 with a custom classification head.

    Architecture changes from vanilla DenseNet121:
    - Remove original 1000-class head
    - Add: GlobalAvgPool → Dropout(0.5) → FC(1024→256) → BN → ReLU →
           Dropout(0.3) → FC(256→num_classes)

    Why this head design?
    - Two FC layers with BN + ReLU allow non-linear combination of features
    - Dropout at two stages aggressively prevents overfitting on small datasets
    - Global Average Pooling before FC reduces parameters vs flattening

    Args:
        num_classes: 2 for binary (heart disease yes/no)
        pretrained: Load ImageNet weights (True) or random init (False)
    Returns:
        nn.Module ready for training or inference
    """
    if pretrained:
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=weights)
    else:
        model = models.densenet121(weights=None)

    # Replace classifier with custom head
    in_features = model.classifier.in_features  # 1024 for DenseNet121
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    return model


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(model_path: str, num_classes: int = 2) -> nn.Module:
    """
    Load model weights from a .pth file.

    Why we use torch.load with map_location?
    The model might have been trained on GPU but we want to run inference
    on CPU. map_location=DEVICE handles this automatically.

    Args:
        model_path: Path to .pth weights file
        num_classes: Must match what the model was trained with
    Returns:
        model in eval() mode, moved to DEVICE
    """
    model = build_model(num_classes=num_classes, pretrained=False)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        # Handle both raw state_dict and checkpoint dicts
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        print(f"[Predictor] Loaded weights from {model_path}")
    else:
        print(f"[Predictor] WARNING: No weights found at {model_path}. Using demo mode.")

    model.to(DEVICE)
    model.eval()
    return model


# ─── HuggingFace Zero-Shot Models ────────────────────────────────────────────

def load_zero_shot_model():
    """
    Loads OpenAI's CLIP model for zero-shot medical image classification.
    """
    print("[Predictor] Downloading/Loading HuggingFace CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()
    print("[Predictor] CLIP model loaded successfully.")
    return model, processor


def zero_shot_predict(image: Image.Image, image_type: str, model, processor) -> dict:
    """
    Runs zero-shot inference using CLIP on the provided image and type.
    """
    if image_type == "ECG/EKG Pattern":
        texts = [
            "ECG of normal sinus rhythm",
            "ECG of atrial fibrillation",
            "ECG of myocardial infarction",
            "ECG of arrhythmia"
        ]
        positive_keywords = ["fibrillation", "infarction", "arrhythmia"]
    elif image_type == "Other Medical Record":
        texts = [
            "Routine medical record",
            "Critical attention required medical document"
        ]
        positive_keywords = ["attention", "critical"]
    elif image_type == "CT Scan":
        texts = [
            "Normal chest CT scan",
            "Chest CT scan with pulmonary embolism",
            "Chest CT scan with lung nodule",
            "Chest CT scan with aortic aneurysm"
        ]
        positive_keywords = ["pulmonary embolism", "lung nodule", "aortic aneurysm"]
    elif image_type == "Echocardiogram":
        texts = [
            "Normal echocardiogram",
            "Echocardiogram with valve regurgitation",
            "Echocardiogram with reduced ejection fraction"
        ]
        positive_keywords = ["valve regurgitation", "reduced ejection fraction"]
    else: # Chest X-Ray
        texts = [
            "Normal clear chest x-ray",
            "Chest x-ray with cardiomegaly",
            "Chest x-ray with congestive heart failure",
            "Chest x-ray with coronary artery disease"
        ]
        positive_keywords = ["cardiomegaly", "failure", "disease"]


    # Preprocess and forward pass
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
    
    # Find max
    predicted_idx = int(np.argmax(probs))
    predicted_text = texts[predicted_idx]
    confidence = float(probs[predicted_idx])
    
    # Determine if finding is positive (abnormal)
    is_positive = any(kw in predicted_text for kw in positive_keywords)
    
    # Format a clean label for UI
    if image_type == "ECG/EKG Pattern":
        if "normal" in predicted_text.lower(): label = "Normal Sinus Rhythm"
        elif "fibrillation" in predicted_text.lower(): label = "Atrial Fibrillation"
        elif "infarction" in predicted_text.lower(): label = "Myocardial Infarction"
        else: label = "Abnormal ECG (Potential Arrhythmia)"
    elif image_type == "Other Medical Record":
        if "Routine" in predicted_text: label = "Routine Medical Record"
        else: label = "Attention Required Document"
    elif image_type == "CT Scan":
        if "pulmonary embolism" in predicted_text: label = "Pulmonary Embolism"
        elif "lung nodule" in predicted_text: label = "Lung Nodule"
        elif "aortic aneurysm" in predicted_text: label = "Aortic Aneurysm"
        else: label = "Normal CT Scan"
    elif image_type == "Echocardiogram":
        if "valve regurgitation" in predicted_text: label = "Valve Regurgitation"
        elif "reduced ejection fraction" in predicted_text: label = "Reduced Ejection Fraction"
        else: label = "Normal Echocardiogram"
    else:
        if "Normal" in predicted_text: label = "No Finding"
        elif "Cardiomegaly" in predicted_text: label = "Cardiomegaly"
        elif "Failure" in predicted_text: label = "Congestive Heart Failure"
        elif "Disease" in predicted_text: label = "Coronary Artery Disease"
        else: label = "Heart Disease"

    img_type_map = {
        "ECG/EKG Pattern": "ECG",
        "Other Medical Record": "Record",
        "CT Scan": "CT Scan",
        "Echocardiogram": "Echocardiogram"
    }
    mapped_type = img_type_map.get(image_type, "X-Ray")

    return {
        "prediction": label,
        "is_positive": is_positive,
        "confidence": confidence,
        "confidence_pct": round(confidence * 100, 1),
        "probabilities": {texts[i]: float(probs[i]) for i in range(len(texts))},
        "demo_mode": False,
        "image_type": mapped_type
    }


# ─── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: nn.Module, image_tensor: torch.Tensor) -> dict:
    """
    Run inference and return structured prediction result.

    Test-Time Augmentation (TTA):
    We run inference on the original image + horizontal flip + slight
    brightness variation, then average probabilities. This reduces variance
    and typically improves AUC by 1-2 points without retraining.

    Args:
        model: Loaded DenseNet121 in eval mode
        image_tensor: (1, 3, 224, 224) float tensor
    Returns:
        dict with keys: prediction, confidence, probabilities
    """
    image_tensor = image_tensor.to(DEVICE)

    # Standard forward pass
    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # TTA: horizontal flip
    flipped = torch.flip(image_tensor, dims=[3])
    logits_flip = model(flipped)
    probs_flip = torch.softmax(logits_flip, dim=1).cpu().numpy()[0]

    # Average TTA predictions
    avg_probs = (probs + probs_flip) / 2.0

    predicted_class = int(np.argmax(avg_probs))
    confidence = float(avg_probs[predicted_class])

    return {
        "prediction": BINARY_LABELS[predicted_class],
        "is_positive": predicted_class == 1,
        "confidence": confidence,
        "confidence_pct": round(confidence * 100, 1),
        "probabilities": {label: float(p) for label, p in zip(BINARY_LABELS, avg_probs)},
    }


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────

def get_gradcam(model: nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for the predicted class.

    How Grad-CAM works:
    1. Forward pass through network, saving the last conv layer's feature maps
    2. Backward pass for the predicted class
    3. Compute gradient of class score w.r.t. each feature map channel
    4. Weight each feature map by its global average gradient (importance weight)
    5. ReLU the weighted sum → only highlight positive contributions
    6. Resize to input image size

    Why the last convolutional layer?
    The last conv layer (denseblock4/denselayer16/conv2 in DenseNet121) has
    the highest-level semantic features while still retaining spatial resolution.
    Earlier layers have spatial info but no semantic content.

    Why Grad-CAM over LIME or SHAP?
    Grad-CAM is faster (single forward+backward pass), native to CNNs,
    and produces spatially coherent maps. LIME and SHAP are model-agnostic
    but much slower and produce noisier maps for images.

    Args:
        model: DenseNet121 in eval mode (grad computation re-enabled inside)
        image_tensor: (1, 3, 224, 224) float tensor
    Returns:
        np.ndarray of shape (224, 224), float32, range [0, 1]
    """
    # Try to use torchcam for clean implementation
    try:
        from torchcam.methods import GradCAM
        cam_extractor = GradCAM(model, target_layer="features.denseblock4")
        model.eval()

        image_tensor = image_tensor.to(DEVICE)
        image_tensor.requires_grad_(True)

        out = model(image_tensor)
        predicted_class = out.argmax(dim=1).item()

        activation_map = cam_extractor(predicted_class, out)
        cam = activation_map[0].squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        return cam

    except Exception:
        # Manual Grad-CAM fallback
        return _manual_gradcam(model, image_tensor)


def _manual_gradcam(model: nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    """
    Manual Grad-CAM implementation as fallback when torchcam is unavailable.

    Registers forward and backward hooks on DenseNet121's last dense block
    to capture feature maps and gradients.
    """
    feature_maps = []
    gradients = []

    def forward_hook(module, input, output):
        feature_maps.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Target: last dense layer in denseblock4
    target_layer = model.features.denseblock4

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    image_tensor = image_tensor.to(DEVICE)
    image_tensor.requires_grad_(True)

    # Forward
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backward
    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()

    fh.remove()
    bh.remove()

    if not feature_maps or not gradients:
        return np.zeros((224, 224), dtype=np.float32)

    fmap = feature_maps[0].detach().cpu().numpy()[0]   # (C, H, W)
    grad = gradients[0].detach().cpu().numpy()[0]       # (C, H, W)

    # Global average pool gradients → channel weights
    weights = grad.mean(axis=(1, 2))  # (C,)

    # Weighted combination of feature maps
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)  # ReLU
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (224, 224))
    return cam


def overlay_heatmap(original_img: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """
    Blend Grad-CAM heatmap onto the original X-ray image.

    Why JET colormap?
    JET maps low activation to blue and high activation to red, which is
    visually intuitive for radiologists: red regions = high model attention.
    TURBO is a perceptually uniform alternative if preferred.

    Args:
        original_img: uint8 RGB array (H, W, 3)
        cam: float32 array (H, W) range [0, 1]
        alpha: blend weight for heatmap (0.4 = subtle overlay)
    Returns:
        uint8 RGB array (H, W, 3) with heatmap overlay
    """
    # Resize original to match
    h, w = original_img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Apply colormap
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    overlaid = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
    return overlaid


# ─── Demo / Mock Predictor ────────────────────────────────────────────────────

def demo_predict(img_rgb: np.ndarray) -> dict:
    """
    Mock prediction for demo mode (no trained model required).

    Uses image statistics as a proxy — images with larger bright central
    regions simulate a cardiomegaly pattern. This is NOT clinically valid;
    it's purely for UI demonstration.

    Args:
        img_rgb: uint8 RGB array
    Returns:
        Same structure as predict() output
    """
    import random
    rng = random.Random(img_rgb.mean().astype(int))

    # Simulate model uncertainty realistically
    confidence = rng.uniform(0.72, 0.94)
    is_positive = rng.random() > 0.45

    label = "Heart Disease" if is_positive else "No Finding"
    other_prob = 1.0 - confidence

    return {
        "prediction": label,
        "is_positive": is_positive,
        "confidence": confidence,
        "confidence_pct": round(confidence * 100, 1),
        "probabilities": {
            label: confidence,
            BINARY_LABELS[1 - int(is_positive)]: other_prob,
        },
        "demo_mode": True,
    }


def demo_gradcam(img_rgb: np.ndarray) -> np.ndarray:
    """
    Generate a plausible-looking demo Grad-CAM without a real model.

    Simulates cardiac region attention by creating a Gaussian blob centered
    slightly left of center (where the heart typically appears in PA chest X-rays).
    """
    h, w = 224, 224
    cam = np.zeros((h, w), dtype=np.float32)

    # Heart is typically left-center in frontal X-ray
    cx, cy = int(w * 0.48), int(h * 0.52)
    sigma = 50

    for y in range(h):
        for x in range(w):
            cam[y, x] = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def demo_predict_ecg(img_rgb: np.ndarray) -> dict:
    """
    Mock prediction for ECG/EKG demo mode.
    """
    import random
    rng = random.Random(img_rgb.mean().astype(int) + 10)

    # Simulate model uncertainty realistically
    confidence = rng.uniform(0.75, 0.98)
    is_positive = rng.random() > 0.40

    label = "Abnormal ECG (Potential Arrhythmia/Ischemia)" if is_positive else "Normal Sinus Rhythm"
    other_prob = 1.0 - confidence

    return {
        "prediction": label,
        "is_positive": is_positive,
        "confidence": confidence,
        "confidence_pct": round(confidence * 100, 1),
        "probabilities": {
            label: confidence,
            "Normal Sinus Rhythm" if is_positive else "Abnormal ECG": other_prob,
        },
        "demo_mode": True,
        "image_type": "ECG",
    }

def demo_gradcam_ecg(img_rgb: np.ndarray) -> np.ndarray:
    """
    Generate a plausible-looking demo Grad-CAM for ECG.
    Focuses on horizontal bands where ECG traces are typically found.
    """
    h, w = 224, 224
    cam = np.zeros((h, w), dtype=np.float32)

    # create horizontal heat waves
    for y in range(h):
        for x in range(w):
            cam[y, x] = np.sin(x/10.0) * np.exp(-((y - h/2.5)**2) / 1000)
    
    cam = np.abs(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def demo_predict_record(img_rgb: np.ndarray) -> dict:
    """
    Mock prediction for General Medical Record demo mode.
    """
    import random
    rng = random.Random(img_rgb.mean().astype(int) + 20)

    # Simulate model uncertainty realistically
    confidence = rng.uniform(0.80, 0.99)
    is_positive = rng.random() > 0.50

    label = "Significant Medical Finding Detected" if is_positive else "Routine Medical Record"
    other_prob = 1.0 - confidence

    return {
        "prediction": label,
        "is_positive": is_positive,
        "confidence": confidence,
        "confidence_pct": round(confidence * 100, 1),
        "probabilities": {
            label: confidence,
            "Routine Medical Record" if is_positive else "Significant Medical Finding Detected": other_prob,
        },
        "demo_mode": True,
        "image_type": "Record",
    }

def demo_gradcam_record(img_rgb: np.ndarray) -> np.ndarray:
    """
    Generate a plausible-looking demo Grad-CAM for medical records.
    Simulates text-like highlights.
    """
    h, w = 224, 224
    import cv2
    cam = np.random.rand(h, w).astype(np.float32)
    cam = cv2.GaussianBlur(cam, (25, 25), 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def demo_gradcam_ct(img_rgb: np.ndarray) -> np.ndarray:
    """
    Generate a plausible-looking demo Grad-CAM for CT Scans.
    Focuses on dual lung node circular spots.
    """
    h, w = 224, 224
    cam = np.zeros((h, w), dtype=np.float32)

    # Simulated lung nodules logic roughly in middle-left and middle-right
    cx1, cy1 = int(w * 0.35), int(h * 0.45)
    cx2, cy2 = int(w * 0.65), int(h * 0.55)
    sigma = 20

    for y in range(h):
        for x in range(w):
            v1 = np.exp(-((x - cx1)**2 + (y - cy1)**2) / (2 * sigma**2))
            v2 = np.exp(-((x - cx2)**2 + (y - cy2)**2) / (2 * (sigma*1.2)**2))
            cam[y, x] = max(v1, v2)

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def demo_gradcam_echo(img_rgb: np.ndarray) -> np.ndarray:
    """
    Generate a plausible-looking demo Grad-CAM for Echocardiograms.
    Focuses on a valve-like structure (diagonal flow).
    """
    h, w = 224, 224
    cam = np.zeros((h, w), dtype=np.float32)

    # diagonal jet-like flow
    for y in range(h):
        for x in range(w):
            dist_to_line = abs((x - w/2) - (y - h/2))
            cam[y, x] = np.exp(-(dist_to_line**2) / 800) * np.exp(-((x - w/2)**2 + (y - h/2)**2)/5000)
    
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam
