import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pymongo import MongoClient

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================
# CONNECT TO MONGODB
# =========================================

print("Connecting to MongoDB...")

client = MongoClient("mongodb://localhost:27017/")

db = client["sentiment_analysis"]

collection = db["balanced_sentiment_dataset"]

# =========================================
# FETCH DATA
# =========================================

print("Fetching dataset from MongoDB...")

data = list(collection.find())

# Convert MongoDB data to DataFrame
df = pd.DataFrame(data)

# =========================================
# REMOVE MONGODB OBJECT ID
# =========================================

if '_id' in df.columns:
    df.drop('_id', axis=1, inplace=True)

# =========================================
# CHECK COLUMNS
# =========================================

print(df.columns)

# Example expected:
# text | sentiment

# =========================================
# INPUT + OUTPUT
# =========================================

X = df['text']
y = df['sentiment']

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================
# LOAD SBERT TRANSFORMER
# =========================================

print("Loading SBERT Transformer...")

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================================
# GENERATE EMBEDDINGS
# =========================================

print("Generating training embeddings...")

X_train_embeddings = sbert_model.encode(
    X_train.tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Generating testing embeddings...")

X_test_embeddings = sbert_model.encode(
    X_test.tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

# =========================================
# TRAIN SGD HINGE LOSS SVM
# =========================================

print("Training SGD Hinge Loss SVM...")

svm_model = SGDClassifier(
    loss='hinge',
    penalty='l2',
    alpha=1e-4,
    max_iter=2000,
    tol=1e-3,
    random_state=42,
    n_jobs=-1
)

svm_model.fit(
    X_train_embeddings,
    y_train
)

# =========================================
# PREDICTIONS
# =========================================

print("Making predictions...")

y_pred = svm_model.predict(
    X_test_embeddings
)

# =========================================
# EVALUATION
# =========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n================================")
print(f"Accuracy: {accuracy:.4f}")
print("================================\n")

print("Classification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =========================================
# CONFUSION MATRIX
# =========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("SBERT + SGD Hinge Loss")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "SBERT_SVM_Confusion_Matrix.png"
)

plt.show()

# =========================================
# SAVE MODELS
# =========================================

joblib.dump(
    svm_model,
    "sbert_svm_model.pkl"
)

print("\nSVM model saved successfully!")

# Save SBERT model

sbert_model.save(
    "sbert_transformer"
)

print("SBERT transformer saved successfully!")