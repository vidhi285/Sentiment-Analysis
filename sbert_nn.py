# ============================================================
# SBERT + NEURAL NETWORK SENTIMENT ANALYSIS
# MONGODB + PYTORCH + EARLY STOPPING
# ============================================================

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt

from pymongo import MongoClient

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sentence_transformers import SentenceTransformer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# DEVICE CONFIGURATION
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing Device: {device}")

# ============================================================
# CONNECT TO MONGODB
# ============================================================

print("\nConnecting to MongoDB...\n")

client = MongoClient("mongodb://localhost:27017/")

db = client["studiobasedproject"]

collection = db["linear_svm"]

print("MongoDB Connected Successfully")

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading Dataset from MongoDB...\n")

data = list(collection.find({}, {
    "clean_text": 1,
    "sentiment": 1,
    "_id": 0
}))

df = pd.DataFrame(data)

print("Dataset Loaded Successfully")

print(df.head())

# ============================================================
# SAMPLE DATASET
# IMPORTANT FOR 8GB RAM
# ============================================================

df = df.sample(50000, random_state=42)

print(f"\nDataset Size After Sampling: {len(df)}")

# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text.strip()

df["clean_text"] = df["clean_text"].apply(clean_text)

# ============================================================
# FEATURES & LABELS
# ============================================================

texts = df["clean_text"].tolist()

labels = df["sentiment"].values

# ============================================================
# LOAD SBERT MODEL
# ============================================================

print("\nLoading SBERT Model...\n")

sbert_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

print("SBERT Model Loaded")

# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating SBERT Embeddings...\n")

embeddings = sbert_model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True
)

print("Embeddings Shape :", embeddings.shape)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    embeddings,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ============================================================
# CUSTOM DATASET
# ============================================================

class SentimentDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)

        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

# ============================================================
# DATALOADERS
# ============================================================

train_dataset = SentimentDataset(X_train, y_train)

test_dataset = SentimentDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16
)

# ============================================================
# NEURAL NETWORK MODEL
# ============================================================

class NeuralNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(384, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 2)

        )

    def forward(self, x):

        return self.network(x)

model = NeuralNetwork().to(device)

# ============================================================
# LOSS & OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2e-5
)

# ============================================================
# EARLY STOPPING VARIABLES
# ============================================================

best_loss = float('inf')

patience = 3

counter = 0

epochs = 10

# ============================================================
# TRAINING LOOP
# ============================================================

print("\nTraining Neural Network...\n")

train_accuracies = []

test_accuracies = []

for epoch in range(epochs):

    model.train()

    total_loss = 0

    correct = 0

    total = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += y_batch.size(0)

        correct += (predicted == y_batch).sum().item()

    train_accuracy = correct / total

    train_accuracies.append(train_accuracy)

    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    correct = 0

    total = 0

    val_loss = 0

    all_preds = []

    all_labels = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(device)

            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += y_batch.size(0)

            correct += (predicted == y_batch).sum().item()

            all_preds.extend(predicted.cpu().numpy())

            all_labels.extend(y_batch.cpu().numpy())

    test_accuracy = correct / total

    test_accuracies.append(test_accuracy)

    print(f"\nEpoch {epoch+1}/{epochs}")

    print(f"Train Accuracy : {train_accuracy:.4f}")

    print(f"Test Accuracy  : {test_accuracy:.4f}")

    print(f"Validation Loss: {val_loss:.4f}")

    # =====================================================
    # EARLY STOPPING
    # =====================================================

    if val_loss < best_loss:

        best_loss = val_loss

        counter = 0

        torch.save(
            model.state_dict(),
            "best_sbert_model.pth"
        )

    else:

        counter += 1

        print(f"Early Stopping Counter: {counter}")

        if counter >= patience:

            print("\nEarly Stopping Triggered!")

            break

# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n==============================")

print("FINAL EVALUATION")

print("==============================")

print("\nClassification Report:\n")

print(classification_report(
    all_labels,
    all_preds
))

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_preds
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("SBERT + Neural Network")

plt.show()

# ============================================================
# SAVE SBERT MODEL
# ============================================================

joblib.dump(
    sbert_model,
    "sbert_encoder.pkl"
)

print("\nSBERT Encoder Saved")

# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n==============================")

print("SAMPLE PREDICTIONS")

print("==============================")

samples = [
    "I absolutely love this app",
    "Worst customer service ever",
    "Amazing movie",
    "I hate this product"
]

sample_embeddings = sbert_model.encode(samples)

sample_tensor = torch.tensor(
    sample_embeddings,
    dtype=torch.float32
).to(device)

model.eval()

with torch.no_grad():

    outputs = model(sample_tensor)

    _, predictions = torch.max(outputs, 1)

for text, pred in zip(samples, predictions):

    sentiment = (
        "Positive"
        if pred.item() == 1
        else "Negative"
    )

    print(f"\nText : {text}")

    print(f"Predicted Sentiment : {sentiment}")

print("\nSBERT + Neural Network Training Completed")