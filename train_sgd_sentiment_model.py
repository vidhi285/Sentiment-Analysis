import pandas as pd
import numpy as np
import joblib
import warnings

from pymongo import MongoClient

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.pipeline import Pipeline

from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm

warnings.filterwarnings("ignore")

# =====================================================
# CONFIGURATION
# =====================================================

MONGO_URI = "mongodb://localhost:27017/"

DATABASE_NAME = "sentimentDB"

COLLECTION_NAME = "sentiments"

SAMPLE_SIZE = 500000
# Start smaller first for stability

MAX_FEATURES = 30000

TEST_SIZE = 0.2

RANDOM_STATE = 42

# =====================================================
# CONNECT MONGODB
# =====================================================

print("\nConnecting MongoDB...")

client = MongoClient(MONGO_URI)

db = client[DATABASE_NAME]

collection = db[COLLECTION_NAME]

print("MongoDB Connected!")

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading dataset from MongoDB...")

cursor = collection.find(
    {},
    {
        "_id": 0
    }
)

data = list(cursor)

df = pd.DataFrame(data)

print(f"\nOriginal Shape: {df.shape}")

# =====================================================
# OPTIONAL SAMPLING
# =====================================================

# IMPORTANT:
# Start with smaller sample first
# Later you can increase gradually

if SAMPLE_SIZE < len(df):
    df = df.sample(
        SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

print(f"\nTraining Shape: {df.shape}")

# =====================================================
# FEATURES & LABELS
# =====================================================

X = df["clean_text"].astype(str)

y = df["sentiment"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# =====================================================
# CLASS WEIGHTS
# =====================================================

# Prevent bias / overfitting

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {
    0: class_weights[0],
    1: class_weights[1]
}

print("\nClass Weights:", class_weight_dict)

# =====================================================
# PIPELINE
# =====================================================

print("\nBuilding TF-IDF + SGD Pipeline...")

pipeline = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(

            max_features=MAX_FEATURES,

            ngram_range=(1,2),

            sublinear_tf=True,

            min_df=5,

            max_df=0.95,

            strip_accents='unicode'
        )
    ),

    (
        "sgd",

        SGDClassifier(

            loss='hinge',
            # Linear SVM style

            penalty='l2',

            alpha=1e-5,
            # Regularization

            max_iter=1000,

            tol=1e-3,

            shuffle=True,

            random_state=RANDOM_STATE,

            learning_rate='optimal',

            early_stopping=True,

            validation_fraction=0.1,

            n_iter_no_change=5,

            class_weight=class_weight_dict
        )
    )
])

# =====================================================
# TRAIN MODEL
# =====================================================

print("\nTraining Model...")

pipeline.fit(X_train, y_train)

print("\nTraining Completed!")

# =====================================================
# PREDICTIONS
# =====================================================

print("\nGenerating Predictions...")

y_pred = pipeline.predict(X_test)

# =====================================================
# EVALUATION
# =====================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 50)

print("MODEL PERFORMANCE")

print("=" * 50)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# =====================================================
# SAVE MODEL
# =====================================================

print("\nSaving model...")

joblib.dump(
    pipeline,
    "tfidf_sgd_sentiment_model.pkl"
)

print("\nModel Saved Successfully!")

# =====================================================
# CLOSE CONNECTION
# =====================================================

client.close()

print("\nMongoDB Connection Closed.")