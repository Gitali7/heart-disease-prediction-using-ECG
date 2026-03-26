"""
model/evaluate.py
==================
Standalone evaluation script to measure model performance.

Run with:
    python model/evaluate.py \
        --model_path ./model/heart_xray_model.pth \
        --data_dir ./datasets/images \
        --labels_file ./datasets/Data_Entry_2017.csv

Outputs:
    - AUC-ROC score
    - F1 score
    - Accuracy
    - Confusion matrix (saved as PNG)
    - ROC curve plot (saved as PNG)
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction.predictor import load_model, DEVICE
from model.train import ChestXRayDataset


def plot_roc_curve(fpr, tpr, auc, save_path="roc_curve.png"):
    """Plot and save ROC curve."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    ax.plot(fpr, tpr, color="#dc2626", lw=2.5, label=f"AUC-ROC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#334155", lw=1, linestyle="--")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#dc2626")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", color="#94a3b8", fontsize=12)
    ax.set_ylabel("True Positive Rate", color="#94a3b8", fontsize=12)
    ax.set_title("ROC Curve — Heart Disease Detection", color="#e2e8f0", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#64748b")
    ax.spines[:].set_color("#1e293b")
    ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155",
              labelcolor="#e2e8f0", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[Evaluate] ROC curve saved to {save_path}")


def plot_confusion_matrix(cm, save_path="confusion_matrix.png"):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    im = ax.imshow(cm, cmap="YlOrRd", vmin=0)
    plt.colorbar(im, ax=ax)

    labels = ["No Finding", "Heart Disease"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color="#94a3b8", fontsize=11)
    ax.set_yticklabels(labels, color="#94a3b8", fontsize=11)
    ax.set_xlabel("Predicted", color="#94a3b8", fontsize=12)
    ax.set_ylabel("Actual", color="#94a3b8", fontsize=12)
    ax.set_title("Confusion Matrix", color="#e2e8f0", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#64748b")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[Evaluate] Confusion matrix saved to {save_path}")


def main(args):
    print(f"[Evaluate] Loading model from {args.model_path}")
    model = load_model(args.model_path)

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = ChestXRayDataset(
        args.data_dir, args.labels_file,
        transform=val_transform, train=False
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"[Evaluate] Evaluating on {len(val_dataset)} samples...")

    all_probs, all_preds, all_labels = [], [], []
    model.eval()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    auc   = roc_auc_score(all_labels, all_probs)
    f1    = f1_score(all_labels, all_preds, zero_division=0)
    acc   = accuracy_score(all_labels, all_preds)
    cm    = confusion_matrix(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    print("\n" + "="*50)
    print("  EVALUATION RESULTS")
    print("="*50)
    print(f"  AUC-ROC Score : {auc:.4f}")
    print(f"  F1 Score      : {f1:.4f}")
    print(f"  Accuracy      : {acc:.4f} ({acc*100:.1f}%)")
    print("="*50)
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=["No Finding", "Heart Disease"]))

    # Save plots
    plot_roc_curve(fpr, tpr, auc, save_path="roc_curve.png")
    plot_confusion_matrix(cm, save_path="confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--labels_file", required=True)
    args = parser.parse_args()
    main(args)
