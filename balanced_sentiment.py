# import pandas as pd
# import re
# import nltk
# #remove common useless words like is, are, the, etc.
# from nltk.corpus import stopwords
# #reduce words to their base form (e.g., "running" → "run"). 
# from nltk.stem import WordNetLemmatizer
# #Resample is used to balance classes to avoid bias in model training.
# from sklearn.utils import resample

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt') # for tokenization

# # Dataset Path
# file_path = r"C:\Users\HP\Documents\Sentiment Analysis Project\dataset\training.1600000.processed.noemoticon.csv"

# # Column names
# columns = ["sentiment","id","date","flag","user","text"]

# # Load dataset & latin-1 is for handling special characters in comments
# df = pd.read_csv(file_path, encoding='latin-1', names=columns)

# print("Original Dataset Shape:", df.shape)

# # Remove duplicates
# df = df.drop_duplicates()

# # Drop unnecessary columns
# df = df.drop(columns=["id","date","flag","user"])

# # Text Cleaning
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# def clean_text(text):
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text)
#     text = re.sub(r'@\w+', '', text)
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     text = text.lower()

#     words = nltk.word_tokenize(text) # Tokenize text into words & Converts sentence -> words
#     words = [word for word in words if word not in stop_words] # Remove stop words
#     words = [lemmatizer.lemmatize(word) for word in words] # Lemmatize words to their base form (e.g., "running" -> "run")

#     return " ".join(words)

# df["clean_text"] = df["text"].apply(clean_text) # Apply cleaning to entire dataset

# df["sentiment"] = df["sentiment"].replace(4,1)

# print("\nBefore Balancing:")
# print(df['sentiment'].value_counts())

# # Separate classes
# df_neg = df[df['sentiment'] == 0]
# df_pos = df[df['sentiment'] == 1]

# # Duplicate positive sample to match negative sample count
# df_pos_upsampled = resample(
#     df_pos,
#     replace=True,
#     n_samples=len(df_neg),
#     random_state=42
# )

# df_balanced = pd.concat([df_neg, df_pos_upsampled]) # Combine
# df_balanced = df_balanced.sample(frac=1, random_state=42) # Shuffle dataset

# print("\nAfter Balancing:")
# print(df_balanced['sentiment'].value_counts())
# df_cleaned = df_balanced[["clean_text","sentiment"]]

# df_cleaned.to_csv("balanced_sentiment_dataset.csv", index=False)

# print("\n Dataset cleaned, balanced, and saved successfully!")











"""
=============================================================
  Balanced Sentiment Dataset Generator
  Author  : Sentiment Analysis Project
  Purpose : Generate ~3M balanced, augmented, unique rows
            from the Sentiment140 Twitter dataset.
=============================================================
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
import re
import sys
import logging

# ── Suppress ALL nltk downloader console output BEFORE importing nlpaug ──────
# nlpaug internally calls nltk.download() on every augment() invocation.
# We silence this by redirecting the nltk downloader's print output
# and also disabling the logging it uses.

import nltk
from unittest.mock import patch                 # patch stdout during downloads

# Run ALL required downloads once, silently, before nlpaug is imported.
_NLTK_PACKAGES = [
    "stopwords",
    "wordnet",
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "omw-1.4",
]

print("Downloading / verifying NLTK packages (one-time) ...")
with open(os.devnull, "w") as _devnull:
    with patch("sys.stdout", _devnull):         # suppress print output
        for _pkg in _NLTK_PACKAGES:
            nltk.download(_pkg, quiet=True)     # quiet=True silences the logger too
print("NLTK packages ready.\n")

# ── Monkey-patch nltk.download so nlpaug can NEVER trigger it again ──────────
# After this point any call to nltk.download() anywhere (including inside nlpaug)
# is a silent no-op.
_original_nltk_download = nltk.download

def _silent_nltk_download(*args, **kwargs):
    """Replacement for nltk.download that does nothing."""
    return True

nltk.download = _silent_nltk_download          # patch BEFORE importing nlpaug

# ── Third-party (import nlpaug AFTER patching) ───────────────────────────────
import pandas as pd
from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer
from nlpaug.augmenter.word import SynonymAug    # safe to import now

# ── Silence noisy loggers from nlpaug / gensim / transformers ────────────────
for _noisy in ("nlpaug", "gensim", "transformers", "nltk"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# =============================================================================
#  CONFIGURATION
# =============================================================================

INPUT_FILE = r"training.1600000.processed.noemoticon.csv"
OUTPUT_FILE = "balanced_sentiment_dataset.csv"

TARGET_TOTAL   = 3_000_000          # desired total rows in final dataset
MIN_WORD_COUNT = 2                  # skip augmented texts shorter than this
PROGRESS_EVERY = 10_000             # print progress every N augmented rows
RANDOM_STATE   = 42

COLUMNS = ["sentiment", "id", "date", "flag", "user", "text"]

# Words that must NEVER be removed by stopword filtering
NEGATION_WORDS = {"not", "no", "never"}

# =============================================================================
#  STEP 0 – CHECK IF OUTPUT ALREADY EXISTS AND IS COMPLETE
# =============================================================================

def is_dataset_complete(path: str, target: int) -> bool:
    """Return True if a valid, balanced dataset already exists."""
    if not os.path.exists(path):
        return False

    print(f"\nExisting file found: {path}")
    try:
        existing = pd.read_csv(path)
    except Exception as exc:
        print(f"Could not read existing file ({exc}). Regenerating.")
        return False

    total  = len(existing)
    counts = existing["sentiment"].value_counts()

    print(f"  Rows      : {total:,}")
    print(f"  Classes   : {dict(counts)}")

    if (
        total >= target
        and len(counts) == 2
        and counts.iloc[0] == counts.iloc[1]
    ):
        print("\n" + "=" * 50)
        print("  DATASET ALREADY PERFECTLY BALANCED")
        print("  NO NEED TO GENERATE AGAIN")
        print("=" * 50)
        return True

    print("\nDataset exists but is incomplete – regenerating.\n")
    return False


if is_dataset_complete(OUTPUT_FILE, TARGET_TOTAL):
    sys.exit(0)

# =============================================================================
#  STEP 1 – LOAD ORIGINAL DATASET
# =============================================================================

print("Loading original dataset ...")
df = pd.read_csv(
    INPUT_FILE,
    encoding="latin-1",
    names=COLUMNS,
)
print(f"  Loaded  : {df.shape[0]:,} rows")

# =============================================================================
#  STEP 2 – INITIAL PREPROCESSING
# =============================================================================

# Keep only what we need
df = df[["sentiment", "text"]].copy()

# Remap labels: 4 → 1
df["sentiment"] = df["sentiment"].replace(4, 1)

# Drop rows with non-binary labels (just in case)
df = df[df["sentiment"].isin([0, 1])]

# Drop duplicate source texts
df = df.drop_duplicates(subset=["text"])
print(f"  After dedup : {df.shape[0]:,} rows")

# =============================================================================
#  STEP 3 – TEXT CLEANING
# =============================================================================

_stop_words  = set(stopwords.words("english")) - NEGATION_WORDS
_lemmatizer  = WordNetLemmatizer()


def clean_text(raw: str) -> str:
    """
    Clean a single tweet:
      - remove URLs, @mentions, non-alphabetic characters
      - lowercase
      - tokenize
      - remove stopwords (keep negations)
      - lemmatize
    Returns the cleaned string, or "" if nothing remains.
    """
    text = str(raw)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)   # URLs
    text = re.sub(r"@\w+",                    " ", text)   # mentions
    text = re.sub(r"[^a-zA-Z\s]",            " ", text)   # non-alpha
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in _stop_words]
    tokens = [_lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens).strip()


print("\nCleaning text (this may take a few minutes) ...")
df["clean_text"] = df["text"].apply(clean_text)

# Remove empty rows that survived cleaning
df = df[df["clean_text"].str.strip().astype(bool)]
print(f"  After cleaning : {df.shape[0]:,} rows")

# =============================================================================
#  STEP 4 – BALANCE CLASSES BEFORE AUGMENTATION
# =============================================================================

df_neg = df[df["sentiment"] == 0]
df_pos = df[df["sentiment"] == 1]

base_size = min(len(df_neg), len(df_pos))

df_neg = df_neg.sample(base_size, random_state=RANDOM_STATE)
df_pos = df_pos.sample(base_size, random_state=RANDOM_STATE)

df_base = pd.concat([df_neg, df_pos]).sample(frac=1, random_state=RANDOM_STATE)
df_base = df_base[["clean_text", "sentiment"]].reset_index(drop=True)

print(f"\nBase balanced dataset : {len(df_base):,} rows")
print(f"  Negative : {(df_base['sentiment']==0).sum():,}")
print(f"  Positive : {(df_base['sentiment']==1).sum():,}")

# =============================================================================
#  STEP 5 – DATA AUGMENTATION
# =============================================================================

# Build the augmenter ONCE (no repeated nltk.download calls possible now)
_aug = SynonymAug(aug_src="wordnet")

augmented_rows: list[dict] = []
seen_texts: set[str]       = set(df_base["clean_text"].tolist())

needed = TARGET_TOTAL - len(df_base)
print(f"\nNeed to augment {needed:,} additional rows to reach {TARGET_TOTAL:,} total.")
print("Generating unique augmented records ...\n")


def augment_text(text: str) -> str | None:
    """
    Attempt synonym augmentation on a text.
    Returns the augmented string, or None if it fails / is too short.
    """
    try:
        result = _aug.augment(text)
        if isinstance(result, list):
            result = result[0]
        result = result.strip()
        if len(result.split()) < MIN_WORD_COUNT:
            return None
        return result
    except Exception:
        return None


# We sample from the base dataframe (with replacement) and augment
# until we have enough unique rows.

base_records = df_base.to_dict("records")   # list of dicts for fast random access
import random
rng = random.Random(RANDOM_STATE)

while len(augmented_rows) < needed:
    record = rng.choice(base_records)
    aug_text = augment_text(record["clean_text"])

    if aug_text is None:
        continue
    if aug_text in seen_texts:
        continue

    seen_texts.add(aug_text)
    augmented_rows.append({
        "clean_text" : aug_text,
        "sentiment"  : record["sentiment"],
    })

    if len(augmented_rows) % PROGRESS_EVERY == 0:
        print(f"  Generated : {len(augmented_rows):,} / {needed:,}")

print(f"\nAugmentation complete – generated {len(augmented_rows):,} rows.")

# =============================================================================
#  STEP 6 – COMBINE, BALANCE AGAIN, SHUFFLE, SAVE
# =============================================================================

df_aug   = pd.DataFrame(augmented_rows)
df_final = pd.concat([df_base, df_aug], ignore_index=True)

# Final balancing pass (augmentation may have produced slightly unequal counts)
df_neg = df_final[df_final["sentiment"] == 0]
df_pos = df_final[df_final["sentiment"] == 1]

final_size = min(len(df_neg), len(df_pos))

df_neg = df_neg.sample(final_size, random_state=RANDOM_STATE)
df_pos = df_pos.sample(final_size, random_state=RANDOM_STATE)

df_final = (
    pd.concat([df_neg, df_pos])
    .sample(frac=1, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

# =============================================================================
#  STEP 7 – SAVE
# =============================================================================

df_final.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 50)
print("  FINAL DATASET CREATED SUCCESSFULLY")
print("=" * 50)
print(f"\n  Shape  : {df_final.shape}")
print(f"  Output : {OUTPUT_FILE}")
print("\n  Class Distribution:")
print(df_final["sentiment"].value_counts().to_string())