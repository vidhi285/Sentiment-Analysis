"""
text_model.py
─────────────────────────────────────────────────────────────
Trains TF-IDF + Logistic Regression on balanced_text_data
from MongoDB, saves the model, and lets you test sentences.
─────────────────────────────────────────────────────────────
MongoDB DB   : sentimentDB
Collection   : balanced_text_data
Required fields: clean_text, sentiment
"""

import pickle
import os
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ─── PATHS ────────────────────────────────────────────────────────────────────
MODEL_SAVE_DIR = r"C:\Users\HP\Documents\Sentiment Analysis Project\saved_models"
VECTORIZER_PATH = os.path.join(MODEL_SAVE_DIR, "tfidf_vectorizer.pkl")
TEXT_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "text_model.pkl")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# ─── STEP 1: FETCH DATA FROM MONGODB ──────────────────────────────────────────
def fetch_text_data():
    """Fetch clean_text + sentiment from MongoDB."""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["sentimentDB"]
    collection = db["balanced_text_data"]

    cursor = collection.find({}, {"_id": 0, "clean_text": 1, "sentiment": 1}) # only fetch needed fields
    df = pd.DataFrame(list(cursor)) # convert cursor to DataFrame
    client.close()

    df = df.dropna(subset=["clean_text", "sentiment"])  # ensure no missing values
    df["clean_text"] = df["clean_text"].astype(str) # ensure text is string type
    print(f"[MongoDB] Loaded {len(df)} text records.") # log how many records were loaded
    return df


# ─── STEP 2: TRAIN ────────────────────────────────────────────────────────────
def train_text_model():
    df = fetch_text_data() # get data from MongoDB

    X = df["clean_text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # TF-IDF vectorizer (unigrams + bigrams, top 5000 features)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000) 
    X_train_vec = vectorizer.fit_transform(X_train) # learn vocab from training data and transform it
    X_test_vec = vectorizer.transform(X_test) # only transform test data (don't learn from it)

    # Logistic Regression classifier
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_vec)) # overall accuracy on test set
    print(f"\n[Text Model] Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, model.predict(X_test_vec))) # detailed metrics for each class

    # Save both vectorizer and model
    with open(VECTORIZER_PATH, "wb") as f: # 
        pickle.dump(vectorizer, f)
    with open(TEXT_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"[Saved] Vectorizer → {VECTORIZER_PATH}") 
    print(f"[Saved] Text model → {TEXT_MODEL_PATH}")

    return vectorizer, model


# ─── STEP 3: LOAD SAVED MODEL ─────────────────────────────────────────────────
def load_text_model():
    """Load the saved vectorizer and model from disk."""
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(TEXT_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("[Loaded] Text model and vectorizer ready.")
    return vectorizer, model


# ─── CLASS LABEL NORMALISER ───────────────────────────────────────────────────
def _normalise_class(cls) -> str:
    """
    Convert whatever label the model stored (int 0/1, string '0'/'1',
    or string 'positive'/'negative'/'neutral') into a consistent string.

    MongoDB dataset uses integers:
        0 → negative
        1 → positive
    Adjust the mapping below if your labels differ.
    """
    INT_MAP = {0: "negative", 1: "positive", 2: "neutral",
               "0": "negative", "1": "positive", "2": "neutral"}
    if cls in INT_MAP:
        return INT_MAP[cls]
    # Already a proper string label
    s = str(cls).strip().lower()
    if s in ("positive", "negative", "neutral"):
        return s
    return "neutral"  # safe fallback


# ─── STEP 4: PREDICT (used by main.py) ────────────────────────────────────────
def predict_text_sentiment(text: str, vectorizer, model) -> dict:
    """
    Returns a dict with keys: positive, negative, neutral
    (probabilities, sum = 1.0)
    """
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for cls, prob in zip(classes, probs):
        label = _normalise_class(cls)
        scores[label] += float(prob)   # += handles duplicate mappings safely

    return scores


# ─── INTERACTIVE TESTER ───────────────────────────────────────────────────────
def interactive_test(vectorizer, model):
    print("\n" + "═" * 55)
    print("  TEXT SENTIMENT TESTER")
    print("  Type a sentence and press Enter.")
    print("  Type 'quit' to exit.")
    print("═" * 55)

    while True:
        sentence = input("\nEnter sentence: ").strip()
        if sentence.lower() == "quit":
            print("Exiting text tester.")
            break
        if not sentence:
            continue

        scores = predict_text_sentiment(sentence, vectorizer, model)
        final = max(scores, key=scores.get)

        print(f"  Positive   : {scores['positive'] * 100:.2f}%")
        print(f"  Negative   : {scores['negative'] * 100:.2f}%")
        print(f"  Neutral    : {scores['neutral'] * 100:.2f}%")
        print(f"  ➤ Predicted: {final.upper()}  (confidence {scores[final] * 100:.2f}%)")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Train if models don't exist yet, otherwise load saved ones
    if os.path.exists(TEXT_MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        print("[Info] Saved model found. Loading...")
        vectorizer, model = load_text_model()
    else:
        print("[Info] No saved model found. Training now...")
        vectorizer, model = train_text_model()

    # Interactive testing
    interactive_test(vectorizer, model)