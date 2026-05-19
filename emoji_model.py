# """
# emoji_model.py
# ─────────────────────────────────────────────────────────────
# Trains MobileNetV2 on emoji images fetched from MongoDB,😭😭
# saves the model, and lets you test by typing any emoji.
# ─────────────────────────────────────────────────────────────
# MongoDB DB   : sentimentDB
# Collections  : apple_emoji, samsung_emoji, google_emoji,
#                facebook_emoji, windows_emoji
# Fields       : path (absolute image path on PC), label

# HOW IT WORKS:
#   MongoDB stores the path and label only.
#   This script reads those paths, opens the actual .png files
#   from your PC, preprocesses them (resize to 224×224, 
#   normalize), and trains MobileNetV2 to classify them.
# ─────────────────────────────────────────────────────────────
# """

# import os
# import pickle
# import emoji
# import numpy as np
# from PIL import Image
# from pymongo import MongoClient

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import models, transforms
# from torch.optim import Adam

# # ─── CONFIG ───────────────────────────────────────────────────────────────────
# MODEL_SAVE_DIR = r"C:\Users\HP\Documents\Sentiment Analysis Project\saved_models"
# EMOJI_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "emoji_mobilenet.pth")
# EMOJI_MAP_PATH = os.path.join(MODEL_SAVE_DIR, "emoji_path_map.pkl")

# os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# LABEL_MAP = {"positive": 0, "negative": 1, "neutral": 2}
# INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# IMG_SIZE = 224      # MobileNet input size
# BATCH_SIZE = 32
# EPOCHS = 10
# LEARNING_RATE = 0.001
# NUM_CLASSES = 3

# # Device: use GPU if available
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"[Device] Using: {DEVICE}")


# # ─── STEP 1: FETCH FROM MONGODB ───────────────────────────────────────────────
# def fetch_emoji_data():
#     """
#     Fetches emoji path + label from all 5 collections.
#     Returns:
#         records   : list of {"path": ..., "label": ...}
#         path_map  : dict mapping emoji unicode codepoint string → image path
#                     (built from filename pattern like '1F600.png')
#     """
#     client = MongoClient("mongodb://localhost:27017/")
#     db = client["sentimentDB"]

#     COLLECTIONS = [
#         "apple_emoji",
#         "samsung_emoji",
#         "google_emoji",
#         "facebook_emoji",
#         "windows_emoji",
#     ]

#     all_records = []

#     for col_name in COLLECTIONS:
#         col = db[col_name]
#         docs = list(col.find({}, {"_id": 0, "path": 1, "label": 1}))
#         for doc in docs:
#             if "path" in doc and "label" in doc:
#                 all_records.append(doc)

#     client.close()
#     print(f"[MongoDB] Total emoji records fetched: {len(all_records)}")

#     # ── Build emoji → image path lookup ──────────────────────────────────────
#     # Emoji image files are typically named by Unicode codepoint e.g. "1F600.png"
#     # We build a map: codepoint_string → path   (from apple collection only,
#     # you can change the priority here)
#     path_map = {}
#     for rec in all_records:
#         path = rec["path"]
#         filename = os.path.basename(path)           # e.g. "1F600.png"
#         name_no_ext = os.path.splitext(filename)[0] # e.g. "1F600"
#         codepoint_key = name_no_ext.lower()         # e.g. "1f600"
#         if codepoint_key not in path_map:           # first occurrence wins
#             path_map[codepoint_key] = path

#     return all_records, path_map


# # ─── STEP 2: PYTORCH DATASET ──────────────────────────────────────────────────
# class EmojiDataset(Dataset):
#     """
#     Loads emoji images from disk using paths stored in MongoDB.
#     Applies resize + normalize preprocessing for MobileNet.
#     """

#     def __init__(self, records, transform=None):
#         self.records = records
#         self.transform = transform

#     def __len__(self):
#         return len(self.records)

#     def __getitem__(self, idx):
#         rec = self.records[idx]
#         path = rec["path"]
#         label_str = rec["label"].strip().lower()

#         # Load and convert image
#         try:
#             img = Image.open(path).convert("RGB")
#         except Exception:
#             # If image can't be opened, return a blank image
#             img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))

#         if self.transform:
#             img = self.transform(img)

#         label_idx = LABEL_MAP.get(label_str, 2)  # default neutral
#         return img, label_idx


# def get_transforms():
#     """Preprocessing transforms for MobileNet input."""
#     return transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         # ImageNet mean/std for pretrained MobileNet
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])


# # ─── STEP 3: BUILD MOBILENET MODEL ────────────────────────────────────────────
# def build_model(num_classes=NUM_CLASSES):
#     """
#     Load pretrained MobileNetV2 and replace the classifier head
#     with a new layer for our 3 classes (positive/negative/neutral).
#     """
#     model = models.mobilenet_v2(pretrained=True)

#     # Freeze all base layers — only train the new classifier head
#     for param in model.features.parameters():
#         param.requires_grad = False

#     # Replace final classifier
#     in_features = model.classifier[1].in_features
#     model.classifier = nn.Sequential(
#         nn.Dropout(p=0.2),
#         nn.Linear(in_features, num_classes)
#     )

#     return model.to(DEVICE)


# # ─── STEP 4: TRAIN ────────────────────────────────────────────────────────────
# def train_emoji_model():
#     records, path_map = fetch_emoji_data()

#     # Filter out records where image file doesn't exist on disk
#     valid_records = [r for r in records if os.path.exists(r["path"])]
#     print(f"[Dataset] Valid image records (file exists): {len(valid_records)}")

#     if len(valid_records) == 0:
#         raise FileNotFoundError(
#             "No valid image files found! Check your MongoDB 'path' values "
#             "and make sure the emoji images exist at those locations on your PC."
#         )

#     transform = get_transforms()
#     dataset = EmojiDataset(valid_records, transform=transform)

#     # 80/20 train-validation split
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
#     val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

#     model = build_model()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.classifier.parameters(), lr=LEARNING_RATE)

#     print(f"\n[Training] Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Device: {DEVICE}")
#     print("─" * 50)

#     best_val_acc = 0.0

#     for epoch in range(1, EPOCHS + 1):
#         # ── Train phase ──────────────────────────────
#         model.train()
#         running_loss = 0.0
#         correct = 0

#         for imgs, labels in train_loader:
#             imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
#             optimizer.zero_grad()
#             outputs = model(imgs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * imgs.size(0)
#             correct += (outputs.argmax(1) == labels).sum().item()

#         train_acc = correct / len(train_ds)
#         train_loss = running_loss / len(train_ds)

#         # ── Validation phase ─────────────────────────
#         model.eval()
#         val_correct = 0
#         with torch.no_grad():
#             for imgs, labels in val_loader:
#                 imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
#                 outputs = model(imgs)
#                 val_correct += (outputs.argmax(1) == labels).sum().item()

#         val_acc = val_correct / len(val_ds)

#         print(
#             f"Epoch {epoch:02d}/{EPOCHS}  |  "
#             f"Loss: {train_loss:.4f}  |  "
#             f"Train Acc: {train_acc * 100:.2f}%  |  "
#             f"Val Acc: {val_acc * 100:.2f}%"
#         )

#         # Save best model weights
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), EMOJI_MODEL_PATH)
#             print(f"  ✔ Best model saved (Val Acc: {val_acc * 100:.2f}%)")

#     # Save emoji path map (used during inference in main.py)
#     with open(EMOJI_MAP_PATH, "wb") as f:
#         pickle.dump(path_map, f)

#     print(f"\n[Saved] Emoji model → {EMOJI_MODEL_PATH}")
#     print(f"[Saved] Emoji path map → {EMOJI_MAP_PATH}")
#     print(f"[Done] Best Validation Accuracy: {best_val_acc * 100:.2f}%")

#     return model, path_map


# # ─── STEP 5: LOAD SAVED MODEL ─────────────────────────────────────────────────
# def load_emoji_model():
#     """Load the trained MobileNet weights and path map from disk."""
#     model = build_model()
#     model.load_state_dict(torch.load(EMOJI_MODEL_PATH, map_location=DEVICE))
#     model.eval()

#     with open(EMOJI_MAP_PATH, "rb") as f:
#         path_map = pickle.load(f)

#     print("[Loaded] Emoji MobileNet model and path map ready.")
#     return model, path_map


# # ─── STEP 6: PREDICT A SINGLE EMOJI IMAGE ─────────────────────────────────────
# def emoji_char_to_codepoint_key(emoji_char: str) -> str:
#     """
#     Convert an emoji character to the codepoint string used as dict key.
#     E.g. '😀' → '1f600'
#     For multi-codepoint emojis (like flags), joins with '-'.
#     E.g. '🇮🇳' → '1f1ee-1f1f3'
#     """
#     codepoints = [f"{ord(c):x}" for c in emoji_char if ord(c) > 0xFFFF or ord(c) > 127]
#     if not codepoints:
#         codepoints = [f"{ord(c):x}" for c in emoji_char]
#     return "-".join(codepoints)

# def predict_emoji_sentiment(emoji_char: str, model, path_map: dict) -> dict:
#     """
#     Given a single emoji character, find its image, run MobileNet,
#     return sentiment scores dict: {positive, negative, neutral}
#     """
#     key = emoji_char_to_codepoint_key(emoji_char)

#     # Try exact key, then try just the first part (for compound emojis)
#     img_path = path_map.get(key) or path_map.get(key.split("-")[0])

#     if img_path is None or not os.path.exists(img_path):
#         # Emoji not found in training data — return neutral
#         return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

#     transform = get_transforms()
#     img = Image.open(img_path).convert("RGB")
#     img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # Add batch dimension

#     with torch.no_grad():
#         output = model(img_tensor)
#         probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()

#     scores = {
#         "positive": float(probabilities[LABEL_MAP["positive"]]),
#         "negative": float(probabilities[LABEL_MAP["negative"]]),
#         "neutral":  float(probabilities[LABEL_MAP["neutral"]]),
#     }
#     return scores


# def get_emoji_scores_mobilenet(emojis: list, model, path_map: dict) -> dict:
#     """
#     Aggregate MobileNet predictions across all emojis in a comment.
#     Returns averaged sentiment scores.
#     (This function is called from main.py as a drop-in replacement)
#     """
#     if not emojis:
#         return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

#     total = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
#     for e in emojis:
#         scores = predict_emoji_sentiment(e, model, path_map)
#         for k in total:
#             total[k] += scores[k]

#     count = len(emojis)
#     return {k: v / count for k, v in total.items()}


# # ─── INTERACTIVE TESTER ───────────────────────────────────────────────────────
# def interactive_test(model, path_map):
#     print("\n" + "═" * 55)
#     print("  EMOJI SENTIMENT TESTER (MobileNet)")
#     print("  Type any emoji and press Enter.")
#     print("  You can type multiple emojis at once.")
#     print("  Type 'quit' to exit.")
#     print("═" * 55)

#     while True:
#         user_input = input("\nEnter emoji(s): ").strip()
#         if user_input.lower() == "quit":
#             print("Exiting emoji tester.")
#             break
#         if not user_input:
#             continue

#         # Extract all emojis from input
#         import emoji as emoji_lib
#         detected = [c for c in user_input if c in emoji_lib.EMOJI_DATA]

#         if not detected:
#             print("  [!] No emojis detected in your input.")
#             continue

#         print(f"  Detected emojis: {detected}")

#         for e in detected:
#             scores = predict_emoji_sentiment(e, model, path_map)
#             key = emoji_char_to_codepoint_key(e)
#             final = max(scores, key=scores.get)
#             print(f"\n  Emoji: {e}  (key: {key})")
#             print(f"    Positive : {scores['positive'] * 100:.2f}%")
#             print(f"    Negative : {scores['negative'] * 100:.2f}%")
#             print(f"    Neutral  : {scores['neutral']  * 100:.2f}%")
#             print(f"    ➤ Predicted: {final.upper()}")


# # ─── ENTRY POINT ──────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     if os.path.exists(EMOJI_MODEL_PATH) and os.path.exists(EMOJI_MAP_PATH):
#         print("[Info] Saved emoji model found. Loading...")
#         model, path_map = load_emoji_model()
#     else:
#         print("[Info] No saved emoji model found. Training now...")
#         model, path_map = train_emoji_model()

#     interactive_test(model, path_map)









"""
============================================================
 emoji_model.py
 Trains a MobileNetV2 model on emoji images stored in MongoDB
 Collections: apple_emoji, facebook_emoji, google_emoji,
              samsung_emoji, windows_emoji
 Labels: positive / negative / neutral  →  0 / 1 / 2
============================================================
"""


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pymongo import MongoClient
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
MONGO_URI       = "mongodb://localhost:27017/"
DB_NAME         = "sentimentDB"
COLLECTIONS     = [
    "apple_emoji",
    "facebook_emoji",
    "google_emoji",
    "samsung_emoji",
    "windows_emoji",
]

IMAGE_SIZE      = 224          # MobileNet expected input
BATCH_SIZE      = 32
NUM_EPOCHS      = 10
LEARNING_RATE   = 0.001
NUM_CLASSES     = 3            # positive, negative, neutral
MODEL_SAVE_PATH = "emoji_model.pth"

# Map string labels → integers
LABEL_MAP = {
    "positive": 0,
    "negative": 1,
    "neutral" : 2,
}

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# ─────────────────────────────────────────────
#  STEP 1 — LOAD DATA FROM CSV
# ─────────────────────────────────────────────
def fetch_data_from_mongodb():
    """
    Reads all 5 emoji CSVs and returns combined lists of (image_path, label_string).
    (Function name kept as fetch_data_from_mongodb so main execution is unaffected, 
    but it now reads from local CSVs).
    """
    print("[STEP 1] Loading from CSVs …")
    
    all_paths  = []
    all_labels = []

    csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSVS")

    for collection_name in COLLECTIONS:
        # e.g. "apple_emoji" -> "CSVS/apple_emoji.csv"
        csv_path = os.path.join(csv_dir, f"{collection_name}.csv")
        
        if not os.path.exists(csv_path):
            print(f"   [WARN] {collection_name}.csv not found at {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        
        # Ensure 'path' and 'label' columns exist
        if 'path' in df.columns and 'label' in df.columns:
            paths  = df['path'].tolist()
            labels = df['label'].tolist()
            
            all_paths.extend(paths)
            all_labels.extend(labels)
            print(f"   ✔ {collection_name}: {len(paths)} records loaded")
        else:
            print(f"   [WARN] {collection_name}.csv is missing 'path' or 'label' columns")

    print(f"[STEP 1] Total records fetched: {len(all_paths)}\n")
    return all_paths, all_labels


# ─────────────────────────────────────────────
#  STEP 2 — PREPARE DATA (encode + split)
# ─────────────────────────────────────────────
def prepare_data(all_paths, all_labels):
    """
    - Converts string labels to integers using LABEL_MAP
    - Skips any records with unknown labels or missing files
    - Splits into train / validation sets (80 / 20)
    """
    print("[STEP 2] Preparing data …")

    clean_paths  = []
    clean_labels = []

    for path, label in zip(all_paths, all_labels):
        label_lower = label.strip().lower()

        if label_lower not in LABEL_MAP:
            print(f"   [WARN] Unknown label '{label}' — skipping")
            continue

        # Paths in DB use Windows separators; normalise for current OS
        normalised_path = os.path.normpath(path)

        if not os.path.isfile(normalised_path):
            # Silently skip missing files (common when DB was built on another machine)
            continue

        clean_paths.append(normalised_path)
        clean_labels.append(LABEL_MAP[label_lower])

    print(f"   Valid samples after filtering: {len(clean_paths)}")

    # Train / Validation split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        clean_paths, clean_labels,
        test_size    = 0.20,
        random_state = 42,
        stratify     = clean_labels,   # keep class balance
    )

    print(f"   Train samples      : {len(train_paths)}")
    print(f"   Validation samples : {len(val_paths)}\n")
    return train_paths, val_paths, train_labels, val_labels


# ─────────────────────────────────────────────
#  STEP 3 — DATASET & DATALOADER
# ─────────────────────────────────────────────
class EmojiDataset(Dataset):
    """
    Custom PyTorch Dataset.
    Loads an emoji image from disk, applies transforms,
    and returns (tensor, label) pairs.
    """

    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Load image → RGB (handles grayscale / RGBA PNGs too)
        image = Image.open(self.paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def build_dataloaders(train_paths, val_paths, train_labels, val_labels):
    """
    Defines image transforms and wraps datasets in DataLoaders.
    Training set gets light augmentation; validation set does not.
    """
    print("[STEP 3] Building image pipeline …")

    # Training transforms — resize, augment, normalise
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        # ImageNet mean/std (MobileNet was pretrained on ImageNet)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # Validation transforms — resize and normalise only
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    train_dataset = EmojiDataset(train_paths, train_labels, transform=train_transform)
    val_dataset   = EmojiDataset(val_paths,   val_labels,   transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"   Train batches      : {len(train_loader)}")
    print(f"   Validation batches : {len(val_loader)}\n")
    return train_loader, val_loader


# ─────────────────────────────────────────────
#  STEP 4 — BUILD MOBILENET MODEL
# ─────────────────────────────────────────────
def build_model():
    """
    Loads pretrained MobileNetV2.
    - Freezes all feature-extraction layers
    - Replaces the final classifier head with a 3-class layer
    """
    print("[STEP 4] Loading pretrained MobileNetV2 …")

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze all layers so only the new head is trained (transfer learning)
    for param in model.features.parameters():
        param.requires_grad = False

    # MobileNetV2 classifier is: [Dropout, Linear(1280, 1000)]
    # We replace it with a 3-class head
    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, NUM_CLASSES),
    )

    model = model.to(DEVICE)
    print(f"   Model ready. Classifier head: Linear({in_features}, {NUM_CLASSES})\n")
    return model


# ─────────────────────────────────────────────
#  STEP 5 & 6 — TRAINING + VALIDATION LOOP
# ─────────────────────────────────────────────
def train_model(model, train_loader, val_loader):
    """
    Full training loop with:
      - CrossEntropyLoss
      - Adam optimiser
      - Epoch-level validation accuracy
      - Best model checkpoint saved to emoji_model.pth
    """
    print("[STEP 5 & 6] Starting training …\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    best_val_accuracy = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):

        # ── TRAINING PHASE ──────────────────────────────────
        model.train()
        running_loss    = 0.0
        correct_train   = 0
        total_train     = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

            # Track loss and accuracy
            running_loss  += loss.item() * images.size(0)
            _, predicted   = torch.max(outputs, dim=1)
            correct_train += (predicted == labels).sum().item()
            total_train   += labels.size(0)

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"   Epoch [{epoch}/{NUM_EPOCHS}] "
                      f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        epoch_loss     = running_loss / total_train
        train_accuracy = correct_train / total_train * 100

        # ── VALIDATION PHASE ────────────────────────────────
        model.eval()
        correct_val = 0
        total_val   = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs   = model(images)
                _, predicted = torch.max(outputs, dim=1)

                correct_val += (predicted == labels).sum().item()
                total_val   += labels.size(0)

        val_accuracy = correct_val / total_val * 100

        print(f"\n{'='*60}")
        print(f"  Epoch [{epoch}/{NUM_EPOCHS}] Summary")
        print(f"  Train Loss     : {epoch_loss:.4f}")
        print(f"  Train Accuracy : {train_accuracy:.2f}%")
        print(f"  Val   Accuracy : {val_accuracy:.2f}%")

        # ── STEP 7: SAVE BEST MODEL ──────────────────────────
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✅ Best model saved → {MODEL_SAVE_PATH}")

        print(f"{'='*60}\n")

    print(f"[DONE] Training complete.")
    print(f"[DONE] Best Validation Accuracy : {best_val_accuracy:.2f}%")
    print(f"[DONE] Model saved at           : {MODEL_SAVE_PATH}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: Fetch data from MongoDB
    all_paths, all_labels = fetch_data_from_mongodb()

    # Step 2: Encode labels and split dataset
    train_paths, val_paths, train_labels, val_labels = prepare_data(all_paths, all_labels)

    # Step 3: Build image pipeline and DataLoaders
    train_loader, val_loader = build_dataloaders(train_paths, val_paths, train_labels, val_labels)

    # Step 4: Load pretrained MobileNetV2 with custom head
    model = build_model()

    # Steps 5, 6 & 7: Train, validate, and save best model
    train_model(model, train_loader, val_loader)