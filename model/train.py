"""
model/train.py
===============
Training script for the DenseNet121 heart X-ray classifier.

Run with:
    python model/train.py \
        --data_dir ./datasets/images \
        --labels_file ./datasets/Data_Entry_2017.csv \
        --epochs 30 \
        --batch_size 32 \
        --output_path ./model/heart_xray_model.pth

Why Focal Loss instead of CrossEntropy?
    The NIH ChestX-ray14 dataset is heavily imbalanced:
    ~60% No Finding, ~5% Cardiomegaly. Standard cross-entropy
    loss is dominated by easy negative examples and underfits the
    rare positive class. Focal Loss (Lin et al., RetinaNet, CVPR 2017)
    reduces the weight of easy examples by multiplying CE loss by
    (1-p_t)^gamma, forcing the model to focus on hard misclassified cases.

Why Cosine Annealing LR?
    Cosine annealing periodically reduces and restores the learning rate,
    helping the optimizer escape sharp local minima. It consistently
    outperforms step decay for medical image classification tasks.

Why Label Smoothing?
    X-ray labels in the NIH dataset contain 5–15% label noise (verified
    by radiologist audit studies). Label smoothing (eps=0.1) prevents
    the model from becoming overconfident on noisy labels.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction.predictor import build_model, DEVICE

# ─── Dataset ──────────────────────────────────────────────────────────────────

class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for NIH ChestX-ray14.

    NIH CSV format:
        Image Index, Finding Labels, Follow-up #, Patient ID, ...
    Finding Labels are pipe-separated: "Cardiomegaly|Effusion"

    Cardiac conditions we classify as positive:
        Cardiomegaly, Edema, Consolidation, Pleural_Thickening

    Args:
        data_dir: Directory containing the X-ray images
        labels_file: Path to Data_Entry_2017.csv
        transform: torchvision transforms
        train: If True, use 80% of data; else 20%
    """
    POSITIVE_CONDITIONS = {
        "Cardiomegaly", "Edema", "Consolidation",
        "Pleural_Thickening", "Effusion"
    }

    def __init__(self, data_dir, labels_file, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform

        df = pd.read_csv(labels_file)
        # Create binary label
        df["label"] = df["Finding Labels"].apply(
            lambda x: 1 if any(c in x for c in self.POSITIVE_CONDITIONS) else 0
        )

        # Train/val split
        split = int(len(df) * 0.8)
        if train:
            self.df = df.iloc[:split].reset_index(drop=True)
        else:
            self.df = df.iloc[split:].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["Image Index"])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Return black image if file missing
            img = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row["label"], dtype=torch.long)
        return img, label


# ─── Loss Functions ───────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance in medical datasets.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=2 is the standard value from the original paper.
    alpha=0.25 upweights the positive (disease) class.

    Why gamma=2?
    At gamma=0, Focal Loss == Cross Entropy.
    At gamma=2, easy examples (high confidence correct predictions)
    have ~100x less weight than hard examples, forcing the model to
    learn from difficult cases — exactly what we need for rare disease classes.
    """
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Apply label smoothing
        n_classes = logits.size(1)
        smooth_targets = torch.zeros_like(logits)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1)
        smooth_targets = smooth_targets * (1 - self.label_smoothing) + \
                         self.label_smoothing / n_classes

        ce = -torch.sum(smooth_targets * torch.log_softmax(logits, dim=1), dim=1)
        probs = torch.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting for positive class
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * focal_weight * ce
        return loss.mean()


# ─── Training Loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scheduler=None):
    """
    Single training epoch.

    Gradient accumulation (every 4 steps) allows effective batch size
    of 4x the actual batch size, important when GPU memory is limited.

    Returns:
        avg_loss: float
    """
    model.train()
    total_loss = 0.0
    accumulation_steps = 4
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss = criterion(logits, labels) / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    """
    Evaluate model on validation set.

    Metrics:
    - AUC-ROC: Area under ROC curve, threshold-independent, best metric
      for imbalanced classification. Target: >0.85 for clinical validity.
    - F1 Score: Harmonic mean of precision and recall. Better than accuracy
      for imbalanced datasets.
    - Accuracy: Simple % correct. Less meaningful for imbalanced data.

    Returns:
        dict of metric name → value
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        "auc_roc": roc_auc_score(all_labels, all_probs),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "accuracy": accuracy_score(all_labels, all_preds),
    }


# ─── Main Training Entrypoint ─────────────────────────────────────────────────

def main(args):
    print(f"[Train] Using device: {DEVICE}")
    print(f"[Train] Epochs: {args.epochs}, Batch size: {args.batch_size}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_dataset = ChestXRayDataset(args.data_dir, args.labels_file,
                                     transform=train_transform, train=True)
    val_dataset   = ChestXRayDataset(args.data_dir, args.labels_file,
                                     transform=val_transform, train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"[Train] Training samples: {len(train_dataset)}")
    print(f"[Train] Validation samples: {len(val_dataset)}")

    # Model
    model = build_model(num_classes=2, pretrained=True).to(DEVICE)

    # Only train classifier head for first 5 epochs (feature extraction phase)
    for param in model.features.parameters():
        param.requires_grad = False

    # Loss & optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )

    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_auc = 0.0

    for epoch in range(args.epochs):
        # Unfreeze backbone after epoch 5 for fine-tuning
        if epoch == 5:
            print("[Train] Unfreezing backbone for fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(),
                                    lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(args.epochs - 5) * len(train_loader)
            )

        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler)
        metrics = evaluate(model, val_loader)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"AUC: {metrics['auc_roc']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "auc_roc": best_auc,
            }, args.output_path)
            print(f"  ✓ Saved best model (AUC: {best_auc:.4f})")

    print(f"\n[Train] Training complete. Best AUC-ROC: {best_auc:.4f}")
    print(f"[Train] Model saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Heart X-ray Classifier")
    parser.add_argument("--data_dir",     required=True,  help="Path to X-ray image directory")
    parser.add_argument("--labels_file",  required=True,  help="Path to Data_Entry_2017.csv")
    parser.add_argument("--epochs",       type=int, default=30)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--output_path",  default="./model/heart_xray_model.pth")
    args = parser.parse_args()
    main(args)
