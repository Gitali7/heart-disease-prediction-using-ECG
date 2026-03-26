"""
utils/helpers.py
================
Shared utility functions used across the project.
"""

import os
import sys
import numpy as np
from PIL import Image
import io


def pil_to_numpy(pil_image) -> np.ndarray:
    """Convert PIL Image to uint8 RGB numpy array."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return np.array(pil_image)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert uint8 numpy array to PIL Image."""
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def get_model_path() -> str:
    """Return the default model weights path."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "model", "heart_xray_model.pth")


def model_exists() -> bool:
    """Check if trained model weights file exists."""
    return os.path.exists(get_model_path())


def validate_image_file(file_obj) -> tuple:
    """
    Validate that uploaded file is a valid image.

    Returns:
        (is_valid: bool, error_message: str)
    """
    if file_obj is None:
        return False, "No file uploaded."

    name = getattr(file_obj, "name", "")
    ext = os.path.splitext(name.lower())[1]

    allowed_extensions = {".jpg", ".jpeg", ".png", ".dcm", ".dicom"}
    if ext not in allowed_extensions:
        return False, f"Unsupported file type '{ext}'. Please upload JPG, PNG, or DICOM."

    # Check file size (limit to 50MB)
    data = file_obj.read()
    file_obj.seek(0)
    size_mb = len(data) / (1024 * 1024)
    if size_mb > 50:
        return False, f"File too large ({size_mb:.1f} MB). Maximum allowed: 50 MB."

    return True, ""


def create_sample_xray() -> np.ndarray:
    """
    Generate a synthetic chest X-ray-like image for demonstration.

    Creates a grayscale gradient image that visually resembles a chest X-ray
    with a roughly circular dark region (simulated heart/lung field).

    Returns:
        uint8 RGB numpy array (400, 400, 3)
    """
    h, w = 400, 400
    img = np.zeros((h, w), dtype=np.float32)

    # Background gradient (simulates X-ray exposure)
    for y in range(h):
        for x in range(w):
            img[y, x] = 0.3 + 0.4 * (1 - abs(x - w//2) / (w//2))

    # Lung fields (lighter regions on both sides)
    for y in range(80, 320):
        for x in range(60, 160):
            dist = ((x - 110)**2 + (y - 200)**2) ** 0.5
            img[y, x] += max(0, (100 - dist) / 200)

    for y in range(80, 320):
        for x in range(240, 340):
            dist = ((x - 290)**2 + (y - 200)**2) ** 0.5
            img[y, x] += max(0, (100 - dist) / 200)

    # Heart silhouette (central slightly dark region)
    for y in range(140, 280):
        for x in range(160, 260):
            dist = ((x - 210)**2 * 1.2 + (y - 210)**2) ** 0.5
            if dist < 60:
                img[y, x] = max(0.1, img[y, x] - 0.15)

    img = np.clip(img, 0, 1)
    img_uint8 = (img * 255).astype(np.uint8)

    # Convert to RGB
    rgb = np.stack([img_uint8, img_uint8, img_uint8], axis=2)
    return rgb
