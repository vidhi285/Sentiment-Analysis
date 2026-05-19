"""
════════════════════════════════════════════════════════════════════════════════
  FILE : emoji_model.py
  ROLE : Training pipeline only — run this ONCE to train and save the model

  Pipeline:
    MongoDB → Validate images → Split → Augment → EfficientNet-B0 → Save

  After training completes → run emoji_predict.py for predictions
════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
from pymongo import MongoClient
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    # MongoDB
    MONGO_URI       = "mongodb://localhost:27017/"
    DB_NAME         = "sentimentDB"
    COLLECTION      = "windows_emoji"

    # Model save path — emoji_predict.py will load from here
    MODEL_PATH      = "emoji_model.pth"

    # Training
    NUM_CLASSES     = 3
    IMG_SIZE        = 224
    BATCH_SIZE      = 16
    EPOCHS          = 20
    LR              = 0.0001
    EARLY_STOP_PAT  = 5
    TRAIN_RATIO     = 0.75
    NUM_WORKERS     = 0          # 0 = Windows safe
    DROPOUT         = 0.4

    # Labels
    LABEL2IDX = {"positive": 0, "negative": 1, "neutral": 2}
    IDX2LABEL = {0: "positive", 1: "negative", 2: "neutral"}

    # ImageNet stats
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]


cfg = Config()


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════════════

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'─'*55}")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"{'─'*55}\n")
    return device


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING FROM MONGODB
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """
    Connect to MongoDB, fetch all records, validate image files exist,
    skip missing/corrupted ones silently, return (paths, labels).
    """
    print("╔══════════════════════════════════════════════════╗")
    print("║         LOADING DATA FROM MONGODB               ║")
    print("╚══════════════════════════════════════════════════╝\n")

    try:
        client     = MongoClient(cfg.MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        collection = client[cfg.DB_NAME][cfg.COLLECTION]
    except Exception as e:
        print(f"[ERROR] MongoDB connection failed: {e}")
        sys.exit(1)

    docs          = list(collection.find({}, {"path": 1, "label": 1, "_id": 0}))
    total         = len(docs)
    print(f"  Total MongoDB records : {total}")

    paths, labels = [], []
    skip_missing  = 0
    skip_corrupt  = 0
    skip_label    = 0

    for doc in docs:
        raw_path  = doc.get("path", "").strip()
        raw_label = doc.get("label", "").strip().lower()

        if raw_label not in cfg.LABEL2IDX:
            skip_label += 1
            continue

        img_path = str(Path(raw_path))
        if not os.path.isfile(img_path):
            skip_missing += 1
            continue

        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            skip_corrupt += 1
            continue

        paths.append(img_path)
        labels.append(cfg.LABEL2IDX[raw_label])

    client.close()

    print(f"  Valid images          : {len(paths)}")
    print(f"  Skipped (missing)     : {skip_missing}")
    print(f"  Skipped (corrupted)   : {skip_corrupt}")
    print(f"  Skipped (bad label)   : {skip_label}\n")

    dist = Counter(labels)
    print("  Class distribution:")
    for idx, name in cfg.IDX2LABEL.items():
        count = dist.get(idx, 0)
        bar   = "█" * (count // 5)
        print(f"    {name:<10}: {count:>4}  {bar}")
    print()

    if len(paths) == 0:
        print("[ERROR] No valid samples found.")
        sys.exit(1)

    return paths, labels


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(cfg.MEAN, cfg.STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.MEAN, cfg.STD),
    ])
    return train_tf, val_tf


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM DATASET
# ══════════════════════════════════════════════════════════════════════════════

class EmojiDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (cfg.IMG_SIZE, cfg.IMG_SIZE), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(self.labels[idx], dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(device):
    """
    EfficientNet-B0 with:
    - Early layers frozen (stem + blocks 0-4)
    - Later layers trainable (blocks 5-8)
    - Custom 3-class classifier head
    """
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    # Freeze early layers
    for i, child in enumerate(model.features.children()):
        for param in child.parameters():
            param.requires_grad = (i >= 5)

    # Replace classifier head
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=cfg.DROPOUT),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, cfg.NUM_CLASSES),
    )

    model = model.to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model      : EfficientNet-B0 (ImageNet pretrained)")
    print(f"  Total      : {total:,} params")
    print(f"  Trainable  : {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"  Frozen     : {total-trainable:,} ({100*(total-trainable)/total:.1f}%)\n")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)

    return loss_sum / total, correct / total


def validate_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out  = model(images)
            loss = criterion(out, labels)

            loss_sum += loss.item() * images.size(0)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += labels.size(0)

    return loss_sum / total, correct / total


# ══════════════════════════════════════════════════════════════════════════════
# SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model):
    """
    Save model weights + architecture metadata.
    emoji_predict.py loads this file.
    """
    torch.save({
        "model_state_dict" : model.state_dict(),
        "label2idx"        : cfg.LABEL2IDX,
        "idx2label"        : cfg.IDX2LABEL,
        "num_classes"      : cfg.NUM_CLASSES,
        "img_size"         : cfg.IMG_SIZE,
        "architecture"     : "efficientnet_b0",
        "classifier_head"  : "1280->512->3",
    }, cfg.MODEL_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# FULL TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_training(device):
    print("\n╔══════════════════════════════════════════════════╗")
    print("║           TRAINING PIPELINE START               ║")
    print("╚══════════════════════════════════════════════════╝\n")

    # Load + split
    paths, labels = load_data()
    tr_p, va_p, tr_l, va_l = train_test_split(
        paths, labels,
        test_size=1 - cfg.TRAIN_RATIO,
        stratify=labels,
        random_state=SEED,
    )
    print(f"  Train : {len(tr_p)}  |  Val : {len(va_p)}\n")

    # Datasets + loaders
    tr_tf, va_tf = get_transforms()
    tr_loader = DataLoader(
        EmojiDataset(tr_p, tr_l, tr_tf),
        batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    va_loader = DataLoader(
        EmojiDataset(va_p, va_l, va_tf),
        batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    # Model + optimizer
    model     = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=1e-6
    )

    # Training loop
    best_val_acc = 0.0
    no_improve   = 0

    print(f"{'─'*65}")
    print(f"  {'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}  Note")
    print(f"{'─'*65}")

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, tr_loader, criterion, optimizer, device)
        va_loss, va_acc = validate_epoch(model, va_loader, criterion, device)
        scheduler.step()

        note = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            no_improve   = 0
            save_model(model)
            note = "★ saved"
        else:
            no_improve += 1

        print(
            f"  {epoch:>3}  {tr_loss:>8.4f}  {tr_acc*100:>6.2f}%"
            f"  {va_loss:>8.4f}  {va_acc*100:>6.2f}%  {note}"
            f"  [{time.time()-t0:.1f}s]"
        )

        if no_improve >= cfg.EARLY_STOP_PAT:
            print(f"\n  Early stop after {cfg.EARLY_STOP_PAT} epochs with no improvement.")
            break

    print(f"{'─'*65}")
    print(f"\n  ✅ Training done!  Best Val Accuracy : {best_val_acc*100:.2f}%")
    print(f"  Model saved → {cfg.MODEL_PATH}")
    print(f"\n  Now run:  python emoji_predict.py\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def print_banner():
    print("\n" + "═"*55)
    print("   EMOJI SENTIMENT — TRAINING SCRIPT")
    print("   EfficientNet-B0  |  PyTorch  |  MongoDB")
    print("═"*55)


def main():
    print_banner()
    device = get_device()

    if os.path.isfile(cfg.MODEL_PATH):
        print(f"  ⚠️  A trained model already exists: {cfg.MODEL_PATH}")
        print("  Retraining will overwrite it.\n")
        ans = input("  Continue and retrain? (y/n): ").strip().lower()
        if ans not in ("y", "yes"):
            print("  Cancelled. Run emoji_predict.py to use existing model.\n")
            sys.exit(0)

    run_training(device)


if __name__ == "__main__":
    main()