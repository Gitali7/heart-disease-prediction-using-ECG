"""
preprocessing/image_processing.py
==================================
All image loading, preprocessing, and augmentation functions.

Why each function exists:
- load_image(): Unified loader for JPG/PNG/DICOM formats
- apply_clahe(): CLAHE (Contrast Limited Adaptive Histogram Equalization)
  dramatically improves low-contrast regions in X-rays, which are often
  underexposed or have poor tissue differentiation. Standard histogram
  equalization distorts the whole image; CLAHE works tile-by-tile.
- normalize_image(): DenseNet121 was pretrained on ImageNet, so it expects
  inputs normalized to ImageNet statistics (mean=[0.485,0.456,0.406],
  std=[0.229,0.224,0.225]). Skipping this drops accuracy by ~5-8%.
- preprocess_for_model(): Combines all steps into a single pipeline call.
- denormalize_for_display(): Reverses normalization before overlay so the
  Grad-CAM heatmap is applied to the human-readable image.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import io

# ImageNet normalization stats — required for pretrained DenseNet121
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224


def load_image(source) -> np.ndarray:
    """
    Load an image from:
      - file path (str)
      - file-like object / BytesIO (from Streamlit uploader)
      - DICOM file path (str ending in .dcm)

    Returns:
        numpy array of shape (H, W, 3), dtype uint8, RGB channel order.

    Why RGB not grayscale?
        Pretrained DenseNet121 expects 3-channel input. We replicate the
        grayscale X-ray across all 3 channels so we can leverage ImageNet
        pretrained weights without architectural changes.
    """
    # --- Handle Streamlit UploadedFile or BytesIO ---
    if hasattr(source, "read"):
        data = source.read()
        source.seek(0)  # Reset for any further reads

        # Try DICOM first
        try:
            import pydicom
            ds = pydicom.dcmread(io.BytesIO(data))
            pixel_array = ds.pixel_array.astype(np.float32)
            # Normalize to 0-255
            pixel_array = ((pixel_array - pixel_array.min()) /
                           (pixel_array.max() - pixel_array.min() + 1e-8) * 255).astype(np.uint8)
            if pixel_array.ndim == 2:
                pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
            return pixel_array
        except Exception:
            pass

        # PIL load for JPG/PNG
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)

    # --- Handle file path ---
    source = str(source)
    if source.lower().endswith(".dcm"):
        try:
            import pydicom
            ds = pydicom.dcmread(source)
            pixel_array = ds.pixel_array.astype(np.float32)
            pixel_array = ((pixel_array - pixel_array.min()) /
                           (pixel_array.max() - pixel_array.min() + 1e-8) * 255).astype(np.uint8)
            if pixel_array.ndim == 2:
                pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
            return pixel_array
        except Exception as e:
            raise ValueError(f"Failed to load DICOM file: {e}")

    img = cv2.imread(source)
    if img is None:
        raise FileNotFoundError(f"Cannot load image from path: {source}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Why CLAHE for X-rays?
    X-rays often have poor local contrast. Standard global histogram
    equalization over-brightens already-bright regions. CLAHE divides
    the image into tiles (8×8 by default) and equalizes each tile
    independently, then stitches them with bilinear interpolation.
    This brings out subtle structural details (cardiac borders,
    vascular markings) that matter for cardiac condition detection.

    clipLimit=2.0 prevents noise amplification.
    tileGridSize=(8,8) balances local vs global contrast.

    Args:
        img_rgb: uint8 RGB array (H, W, 3)
    Returns:
        uint8 RGB array with enhanced local contrast
    """
    # Work in LAB color space — CLAHE is applied only to L (luminance)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def normalize_image(img_rgb: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values using ImageNet mean/std.

    Why these specific values?
    DenseNet121 was trained on ImageNet using these statistics. Applying
    the same normalization aligns the input distribution with what the
    pretrained feature extractors expect, making fine-tuning faster and
    more stable. Without this, training diverges or requires much higher
    learning rate warmup time.

    Returns:
        float32 array normalized to approximately [-2, 2] range.
    """
    img = img_rgb.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    return (img - mean) / std


def preprocess_for_model(img_rgb: np.ndarray) -> torch.Tensor:
    """
    Full preprocessing pipeline: raw RGB → model-ready tensor.

    Pipeline:
        1. Resize to 224×224 (DenseNet expected input)
        2. Apply CLAHE for X-ray contrast enhancement
        3. Normalize with ImageNet statistics
        4. Convert HWC → CHW (PyTorch channel-first format)
        5. Add batch dimension: (C,H,W) → (1,C,H,W)

    Returns:
        torch.FloatTensor of shape (1, 3, 224, 224)
    """
    # Step 1: Resize
    img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE),
                             interpolation=cv2.INTER_LINEAR)

    # Step 2: CLAHE
    img_enhanced = apply_clahe(img_resized)

    # Step 3: Normalize
    img_norm = normalize_image(img_enhanced)

    # Step 4 & 5: HWC → CHW → NCHW
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
    return tensor.float()


def denormalize_for_display(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization.

    Why needed?
    Grad-CAM overlays are applied to the original image for doctor review.
    The normalized tensor values are not human-interpretable. This function
    reverses the normalization so the base image looks like the original X-ray.

    Args:
        tensor: (1, 3, H, W) or (3, H, W) float tensor
    Returns:
        uint8 RGB numpy array (H, W, 3)
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    img = tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def get_training_transforms():
    """
    Training-time augmentation pipeline using albumentations.

    Why these augmentations?
    - HorizontalFlip: Anatomically, the heart is slightly left-of-center,
      but flipping is still valid augmentation for structural features.
    - RandomBrightnessContrast: Simulates different X-ray exposure levels.
    - ShiftScaleRotate: Simulates patient positioning variation (common
      in real-world radiographs).
    - GaussNoise: Simulates detector noise in lower-quality X-rays.
    - CLAHE (in augmentation): Applied probabilistically to further
      vary contrast during training.

    Returns:
        albumentations.Compose transform object
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=10, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    except ImportError:
        # Fallback to torchvision transforms if albumentations not installed
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_validation_transforms():
    """
    Validation/inference transforms — no augmentation, only resize + normalize.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
