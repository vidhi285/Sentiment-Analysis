import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# ==============================================================================
# PHASE 2: SBERT + DEEP LEARNING PIPELINE
# ==============================================================================
# 
# Pipeline Flow:
# MongoDB -> Pandas -> SBERT Embeddings -> Train/Test Split -> 
# Keras DL Model -> Evaluation
#
# Why SBERT instead of TF-IDF?
# - TF-IDF only captures exact word frequencies (lexical matching). 
# - SBERT (Sentence-BERT) captures deep semantic meaning and context.
#   "I am not happy" and "I am sad" have completely different words but 
#   SBERT knows they mean the same thing.
#
# Why Deep Learning?
# - Logistic Regression (used previously) is a linear classifier.
# - Deep Learning (Dense neural networks) can learn complex, non-linear 
#   combinations of SBERT embedding features, leading to higher accuracy.
# ==============================================================================

# --- Configuration ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "sentiment_analysis_db"
COLLECTION_NAME = "tweets_dataset"
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_LIMIT = 0  # Adjust based on RAM. Using a subset for demonstration/speed if needed.
                     # Set to 0 to load everything (beware of RAM limits with 3M rows!)
SAVED_MODEL_DIR = "saved_models/sbert_keras_model.h5"

def fetch_data_from_mongodb(limit=0):
    """Fetches data from MongoDB into a pandas DataFrame."""
    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    print(f"Fetching data from '{COLLECTION_NAME}'...")
    start_time = time.time()
    
    # We project out the MongoDB '_id' field as it's not needed for training
    cursor = collection.find({}, {'_id': 0, 'clean_text': 1, 'sentiment': 1})
    
    if limit > 0:
        cursor = cursor.limit(limit)
        
    df = pd.DataFrame(list(cursor))
    client.close()
    
    print(f"Fetched {len(df)} records in {time.time() - start_time:.2f} seconds.")
    return df

def generate_embeddings(texts):
    """Converts a list of texts into SBERT embeddings."""
    print(f"Loading SBERT model '{MODEL_NAME}'...")
    # Check for GPU
    device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
    print(f"Using device: {device}")
    
    sbert_model = SentenceTransformer(MODEL_NAME, device=device)
    
    print(f"Generating embeddings for {len(texts)} texts...")
    start_time = time.time()
    # show_progress_bar is useful for large datasets
    embeddings = sbert_model.encode(texts, show_progress_bar=True, batch_size=256)
    print(f"Embeddings generated in {time.time() - start_time:.2f} seconds.")
    
    return embeddings, sbert_model

def build_model(input_dim):
    """Builds a Keras Sequential Deep Learning model."""
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid') # Binary classification (0 = Neg, 1 = Pos)
    ])
    
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model

def train_and_evaluate(X, y):
    """Splits data, trains the model, and evaluates it."""
    print("Splitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Building Deep Learning Model...")
    model = build_model(input_dim=X_train.shape[1])
    model.summary()
    
    # Early stopping prevents overfitting if validation loss stops improving
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=512,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("\nEvaluating model on test set...")
    # Predict probabilities, then threshold at 0.5
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model

def save_trained_model(model):
    """Saves the Keras model to disk."""
    os.makedirs(os.path.dirname(SAVED_MODEL_DIR), exist_ok=True)
    model.save(SAVED_MODEL_DIR)
    print(f"\nModel saved successfully at: {SAVED_MODEL_DIR}")

def predict_custom_sentence(sentence, sbert_model, keras_model):
    """End-to-end inference for a single custom sentence."""
    print(f"\nPredicting sentiment for: '{sentence}'")
    emb = sbert_model.encode([sentence])
    prob = keras_model.predict(emb)[0][0]
    
    sentiment = "Positive" if prob > 0.5 else "Negative"
    print(f"Prediction: {sentiment} (Confidence: {prob:.4f})")

if __name__ == "__main__":
    # 1. Fetch from MongoDB
    df = fetch_data_from_mongodb(limit=BATCH_LIMIT) 
    
    if df.empty:
        print("No data found. Did you run Phase 1 (mongodb_ingest.py) first?")
        exit()
        
    texts = df['clean_text'].astype(str).tolist()
    labels = df['sentiment'].values
    
    # 2. Generate Embeddings
    X_embeddings, sbert_encoder = generate_embeddings(texts)
    
    # 3. Train and Evaluate
    trained_model = train_and_evaluate(X_embeddings, labels)
    
    # 4. Save Model
    save_trained_model(trained_model)
    
    # 5. Manual Sentiment Prediction
    while True:
        user_text = input("\nEnter a comment to analyze (type 'exit' to quit): ")

        if user_text.lower() == 'exit':
            print("Exiting sentiment analyzer...")
            break

        predict_custom_sentence(user_text, sbert_encoder, trained_model)
