"""
text_model.py
─────────────────────────────────────────────────────────────
SBERT + Deep Learning Text Sentiment Classifier

Replaces the old TF-IDF + Logistic Regression approach with:
  1. Sentence-BERT (all-MiniLM-L6-v2) for text embeddings
  2. TensorFlow/Keras Dense Neural Network for classification

The SBERT model converts each sentence into a fixed 384-dimensional
dense vector that captures semantic meaning. This is far richer
than the sparse bag-of-words vectors produced by TF-IDF.

A 3-layer neural network then classifies these embeddings into
sentiment categories (positive / negative / neutral).
─────────────────────────────────────────────────────────────
MongoDB DB   : sentimentDB
Collection   : balanced_text_data
Required fields: clean_text, sentiment
─────────────────────────────────────────────────────────────
"""

# ═══════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════

import os
import pickle
import numpy as np
import pandas as pd

# Sentence-BERT — generates dense sentence embeddings
from sentence_transformers import SentenceTransformer

# TensorFlow / Keras — builds and trains the deep learning classifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# scikit-learn — data splitting, label encoding, evaluation metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# SBERT model name — a compact, fast, and accurate sentence encoder
# Outputs 384-dimensional embeddings for each input sentence
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# Directory where trained model artifacts are saved
# Using project-local path so it works on any machine
MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Paths for saved artifacts
KERAS_MODEL_PATH   = os.path.join(MODEL_SAVE_DIR, "sbert_text_model.keras")
LABEL_ENCODER_PATH = os.path.join(MODEL_SAVE_DIR, "sbert_label_encoder.pkl")

# Training hyperparameters
EPOCHS          = 10       # Number of training passes over the dataset
BATCH_SIZE      = 64       # Samples per gradient update
VALIDATION_SPLIT = 0.15    # Fraction of training data used for validation during training
TEST_SIZE       = 0.25     # Fraction of total data held out for final evaluation
RANDOM_STATE    = 42       # For reproducibility


# ═══════════════════════════════════════════════════════════════
#  STEP 1: FETCH DATA FROM MONGODB
# ═══════════════════════════════════════════════════════════════

def fetch_text_data():
    """
    Load the balanced text sentiment dataset from MongoDB.

    The collection should have documents with:
        - clean_text : preprocessed text (string)
        - sentiment  : label (0 = negative, 1 = positive)

    Returns:
        pd.DataFrame with columns: clean_text, sentiment
    """
    from pymongo import MongoClient
    MONGO_URI = "mongodb://localhost:27017/"
    DB_NAME = "sentimentDB"
    COLLECTION_NAME = "balanced_text_data"

    print(f"[MongoDB] Connecting to {MONGO_URI}...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    print(f"[MongoDB] Fetching records from '{DB_NAME}.{COLLECTION_NAME}'...")
    # Fetch all records, returning only the two fields we need (and omitting the _id)
    cursor = collection.find({}, {"_id": 0, "clean_text": 1, "sentiment": 1})
    
    # Convert cursor to DataFrame
    df = pd.DataFrame(list(cursor))

    if df.empty:
        raise ValueError(
            f"No data found in MongoDB ({DB_NAME}.{COLLECTION_NAME}).\n"
            "Please run 'python extend_and_ingest.py' first to populate the database."
        )

    # Drop rows with missing values in either column
    df = df.dropna(subset=["clean_text", "sentiment"])

    # Ensure text column is string type
    df["clean_text"] = df["clean_text"].astype(str)

    print(f"[MongoDB] Loaded {len(df)} text records from database.")
    return df


# ═══════════════════════════════════════════════════════════════
#  STEP 2: TRAIN THE SBERT + DEEP LEARNING MODEL
# ═══════════════════════════════════════════════════════════════

def train_text_model():
    """
    Full training pipeline:
      1. Fetch data from MongoDB
      2. Encode labels (string/int → integer classes)
      3. Generate SBERT embeddings for all texts
      4. Split into train and test sets
      5. Build a Keras Dense Neural Network
      6. Train the model with early stopping
      7. Evaluate with accuracy, classification report, confusion matrix
      8. Save the trained model and label encoder

    Returns:
        (sbert_model, keras_model, label_encoder)
    """
    # ── 2.1  Fetch data ────────────────────────────────────────
    df = fetch_text_data()

    texts      = df["clean_text"].tolist()
    raw_labels = df["sentiment"].tolist()

    # ── 2.2  Encode labels ─────────────────────────────────────
    # LabelEncoder converts labels (could be 0/1, "positive"/"negative", etc.)
    # into consecutive integers starting from 0
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_labels)

    num_classes = len(label_encoder.classes_)
    print(f"\n[Labels] {num_classes} classes detected: {list(label_encoder.classes_)}")

    # ── 2.3  Generate SBERT embeddings ─────────────────────────
    # The SBERT model converts each sentence into a 384-dim dense vector
    # that captures its semantic meaning
    print(f"\n[SBERT] Loading model: {SBERT_MODEL_NAME} ...")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

    print(f"[SBERT] Encoding {len(texts)} sentences (this may take a few minutes) ...")
    X = sbert_model.encode(
        texts,
        show_progress_bar=True,   # Visual progress bar during encoding
        batch_size=256,           # Process 256 sentences at a time for efficiency
    )
    # X is now a numpy array of shape (num_samples, 384)
    print(f"[SBERT] Embeddings shape: {X.shape}")

    # ── 2.4  Train/Test split ──────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,               # Maintain class balance in both splits
    )
    print(f"\n[Split] Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── 2.5  Build Keras Deep Learning model ───────────────────
    # Architecture:
    #   Input  → Dense(256, ReLU)
    #          → Dropout(0.3)      ← prevents overfitting
    #          → Dense(128, ReLU)
    #          → Dense(num_classes, softmax)  ← output probabilities
    embedding_dim = X_train.shape[1]   # 384 for all-MiniLM-L6-v2

    keras_model = Sequential([
        Input(shape=(embedding_dim,)),              # Explicit input shape
        Dense(256, activation="relu"),               # First hidden layer
        Dropout(0.3),                                # Regularisation layer
        Dense(128, activation="relu"),               # Second hidden layer
        Dense(num_classes, activation="softmax"),    # Output layer
    ])

    # Compile the model
    # - Adam: adaptive learning rate optimiser (good default)
    # - sparse_categorical_crossentropy: integer labels, multi-class
    keras_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Print model architecture summary
    print("\n[Model] Architecture:")
    keras_model.summary()

    # ── 2.6  Train with early stopping ─────────────────────────
    # Early stopping monitors validation loss and stops training
    # if it doesn't improve for 3 consecutive epochs
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,   # Revert to the best epoch's weights
        verbose=1,
    )

    print(f"\n[Training] Starting training for up to {EPOCHS} epochs ...")
    print(f"           Batch size: {BATCH_SIZE}")
    print(f"           Validation split: {VALIDATION_SPLIT}")
    print("-" * 60)

    history = keras_model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop],
        verbose=1,                  # Show progress per epoch
    )

    # ── 2.7  Evaluate on test set ──────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    # Get predictions
    y_pred_probs = keras_model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    # Overall accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {acc * 100:.2f}%")

    # Detailed classification report (precision, recall, F1 per class)
    target_names = [str(c) for c in label_encoder.classes_]
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"  {cm}\n")

    # ── 2.8  Save model artifacts ──────────────────────────────
    # Save the Keras model (architecture + weights + optimizer state)
    keras_model.save(KERAS_MODEL_PATH)
    print(f"[Saved] Keras model     -> {KERAS_MODEL_PATH}")

    # Save the LabelEncoder (maps integer predictions back to labels)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"[Saved] Label encoder   -> {LABEL_ENCODER_PATH}")

    return sbert_model, keras_model, label_encoder


# ═══════════════════════════════════════════════════════════════
#  STEP 3: LOAD SAVED MODEL
# ═══════════════════════════════════════════════════════════════

def load_text_model():
    """
    Load all components needed for text sentiment prediction:
      1. SBERT model (auto-downloaded/cached by sentence-transformers)
      2. Trained Keras classifier from disk
      3. LabelEncoder from disk

    Returns:
        (sbert_model, keras_model, label_encoder)
    """
    # Load SBERT model (cached after first download, ~80MB)
    print(f"[Loading] SBERT model: {SBERT_MODEL_NAME} ...")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

    # Load trained Keras classifier
    print(f"[Loading] Keras model: {KERAS_MODEL_PATH} ...")
    keras_model = load_model(KERAS_MODEL_PATH)

    # Load label encoder
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    print("[Loaded] Text model (SBERT + Keras) ready.")
    return sbert_model, keras_model, label_encoder


# ═══════════════════════════════════════════════════════════════
#  CLASS LABEL NORMALISER
# ═══════════════════════════════════════════════════════════════

def _normalise_class(cls) -> str:
    """
    Convert whatever label the model stored (int 0/1, string '0'/'1',
    or string 'positive'/'negative'/'neutral') into a consistent string.

    MongoDB dataset uses integers:
        0 → negative
        1 → positive
    Adjust the mapping below if your labels differ.
    """
    INT_MAP = {
        0: "negative", 1: "positive", 2: "neutral",
        "0": "negative", "1": "positive", "2": "neutral",
    }
    if cls in INT_MAP:
        return INT_MAP[cls]

    # Already a proper string label
    s = str(cls).strip().lower()
    if s in ("positive", "negative", "neutral"):
        return s

    return "neutral"   # safe fallback


# ═══════════════════════════════════════════════════════════════
#  STEP 4: PREDICT SENTIMENT (used by main.py)
# ═══════════════════════════════════════════════════════════════

def predict_text_sentiment(
    text: str,
    sbert_model,
    keras_model,
    label_encoder,
) -> dict:
    """
    Predict sentiment for a single text input.

    Steps:
      1. Convert text → 384-dim SBERT embedding
      2. Feed embedding through Keras model → softmax probabilities
      3. Map each class probability to its label name

    Args:
        text          : The input sentence to classify
        sbert_model   : Loaded SentenceTransformer instance
        keras_model   : Loaded Keras Sequential model
        label_encoder : Loaded LabelEncoder instance

    Returns:
        dict with keys: positive, negative, neutral
        (probabilities that sum to ~1.0)
    """
    # Step 1: Generate SBERT embedding for the input text
    # Returns shape (384,) — we reshape to (1, 384) for the model
    embedding = sbert_model.encode([text])

    # Step 2: Get prediction probabilities from Keras model
    probs = keras_model.predict(embedding, verbose=0)[0]

    # Step 3: Map probabilities to sentiment labels
    # Start with all three keys at 0.0
    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    for idx, prob in enumerate(probs):
        # Get the original label for this index from LabelEncoder
        original_label = label_encoder.inverse_transform([idx])[0]
        # Normalise to standard string ("positive"/"negative"/"neutral")
        label = _normalise_class(original_label)
        # Accumulate probability (handles edge cases safely)
        scores[label] += float(prob)

    return scores


# ═══════════════════════════════════════════════════════════════
#  INTERACTIVE TESTER
# ═══════════════════════════════════════════════════════════════

def interactive_test(sbert_model, keras_model, label_encoder):
    """
    Interactive console tool for testing sentiment predictions.
    Type any sentence and see the model's prediction instantly.
    """
    print("\n" + "=" * 55)
    print("  TEXT SENTIMENT TESTER (SBERT + Deep Learning)")
    print("  Type a sentence and press Enter.")
    print("  Type 'quit' to exit.")
    print("=" * 55)

    while True:
        sentence = input("\nEnter sentence: ").strip()
        if sentence.lower() == "quit":
            print("Exiting text tester.")
            break
        if not sentence:
            continue

        # Get prediction scores
        scores = predict_text_sentiment(sentence, sbert_model, keras_model, label_encoder)
        final  = max(scores, key=scores.get)

        # Display results
        print(f"  Positive   : {scores['positive'] * 100:.2f}%")
        print(f"  Negative   : {scores['negative'] * 100:.2f}%")
        print(f"  Neutral    : {scores['neutral'] * 100:.2f}%")
        print(f"  > Predicted: {final.upper()}  (confidence {scores[final] * 100:.2f}%)")


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Train if saved model doesn't exist yet, otherwise load it
    if os.path.exists(KERAS_MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
        print("[Info] Saved SBERT + Keras model found. Loading ...")
        sbert_model, keras_model, label_encoder = load_text_model()
    else:
        print("[Info] No saved model found. Training now ...")
        sbert_model, keras_model, label_encoder = train_text_model()

    # Launch interactive testing
    interactive_test(sbert_model, keras_model, label_encoder)