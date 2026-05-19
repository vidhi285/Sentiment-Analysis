# ============================================================
# SENTIMENT ANALYSIS USING SVM + TF-IDF + MONGODB
# ============================================================

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import re
import nltk
import joblib
import matplotlib.pyplot as plt

from pymongo import MongoClient

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# CONNECT TO MONGODB
# ============================================================

print("\nConnecting to MongoDB...\n")

client = MongoClient("mongodb://localhost:27017/")

db = client["studiobasedproject"]

collection = db["linear_svm"]

print("MongoDB Connected Successfully")

# ============================================================
# LOAD DATA FROM MONGODB
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
# OPTIONAL: REDUCE DATASET SIZE
# (IMPORTANT FOR 8GB RAM)
# ============================================================

df = df.sample(100000, random_state=42)

print(f"\nDataset Size After Sampling: {len(df)}")

# ============================================================
# TEXT CLEANING
# ============================================================

print("\nCleaning Text...\n")

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove Mentions
    text = re.sub(r"@\w+", "", text)

    # Remove Special Characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    words = text.split()

    # Remove Stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# Apply Cleaning
df["clean_text"] = df["clean_text"].apply(clean_text)

print("Text Cleaning Completed")

# ============================================================
# FEATURES & LABELS
# ============================================================

X = df["clean_text"]

y = df["sentiment"]

# ============================================================
# TF-IDF VECTORIZATION
# ============================================================

print("\nApplying TF-IDF...\n")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_tfidf = vectorizer.fit_transform(X)

print("TF-IDF Shape :", X_tfidf.shape)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", X_train.shape[0])

print("Testing Samples  :", X_test.shape[0])

# ============================================================
# MODEL
# ============================================================

print("\nTraining SVM Model...\n")

model = LinearSVC(
    C=1,
    dual="auto",
    max_iter=3000
)

# ============================================================
# TRAINING
# ============================================================

model.fit(X_train, y_train)

print("Model Training Completed")

# ============================================================
# PREDICTIONS
# ============================================================

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)

# ============================================================
# ACCURACY
# ============================================================

train_accuracy = accuracy_score(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n==============================")

print("MODEL ACCURACY")

print("==============================")

print(f"\nTrain Accuracy : {train_accuracy:.4f}")

print(f"Test Accuracy  : {test_accuracy:.4f}")

# ============================================================
# OVERFITTING CHECK
# ============================================================

difference = train_accuracy - test_accuracy

print("\n==============================")

print("OVERFITTING CHECK")

print("==============================")

print(f"\nAccuracy Difference : {difference:.4f}")

if difference > 0.05:
    print("Warning: Possible Overfitting")
else:
    print("Model Looks Good")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==============================")

print("CLASSIFICATION REPORT")

print("==============================")

print(classification_report(y_test, y_test_pred))

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nGenerating Confusion Matrix...\n")

cm = confusion_matrix(y_test, y_test_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Confusion Matrix - Linear SVM")

plt.show()

# ============================================================
# CROSS VALIDATION
# ============================================================

print("\nPerforming Cross Validation...\n")

cv_scores = cross_val_score(
    model,
    X_tfidf,
    y,
    cv=3
)

print("Cross Validation Scores :", cv_scores)

print("Average CV Accuracy :", cv_scores.mean())

# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving Model...\n")

joblib.dump(model, "svm_model.pkl")

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model Saved Successfully")

# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n==============================")

print("SAMPLE PREDICTIONS")

print("==============================")

samples = [
    "I absolutely love this app",
    "This is the worst experience ever",
    "Amazing service and great support",
    "I hate this product"
]

sample_vector = vectorizer.transform(samples)

predictions = model.predict(sample_vector)

for text, pred in zip(samples, predictions):

    sentiment = "Positive" if pred == 1 else "Negative"

    print(f"\nText : {text}")

    print(f"Predicted Sentiment : {sentiment}")

# ============================================================
# END
# ============================================================

print("\nSVM Sentiment Analysis Completed Successfully")