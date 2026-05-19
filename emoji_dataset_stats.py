"""
emoji_dataset_stats.py
────────────────────────────────────────────────────────────
Checks Windows emoji dataset statistics from MongoDB.

What this script does:
1. Connects to MongoDB
2. Fetches image path + label
3. Checks whether image file exists
4. Counts:
    - positive emojis
    - negative emojis
    - neutral emojis
5. Counts missing/deleted emojis
6. Shows class distribution clearly

Useful when:
- you delete unnecessary neutral object emojis
- you want to verify dataset balance
- you want to see how many files are missing
────────────────────────────────────────────────────────────
"""

import os
from pymongo import MongoClient
from collections import Counter

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MONGO_URI  = "mongodb://localhost:27017/"
DB_NAME    = "sentimentDB"
COLLECTION = "windows_emoji"

# Valid labels
VALID_LABELS = ["positive", "negative", "neutral"]

# ─────────────────────────────────────────────
# CONNECT TO MONGODB
# ─────────────────────────────────────────────
print("\n[STEP 1] Connecting to MongoDB...")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION]

print("Connected successfully.\n")

# ─────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────
print("[STEP 2] Fetching emoji records...")

documents = list(
    collection.find(
        {},
        {
            "path": 1,
            "label": 1,
            "_id": 0
        }
    )
)

print(f"Total MongoDB records found: {len(documents)}\n")

# ─────────────────────────────────────────────
# STATISTICS VARIABLES
# ─────────────────────────────────────────────
existing_counter = Counter()

missing_counter = Counter()

total_existing = 0
total_missing = 0
invalid_labels = 0

# ─────────────────────────────────────────────
# PROCESS DATA
# ─────────────────────────────────────────────
print("[STEP 3] Checking image paths...\n")

for doc in documents:

    # Check required fields
    if "path" not in doc or "label" not in doc:
        continue

    path = os.path.normpath(doc["path"])
    label = doc["label"].strip().lower()

    # Skip invalid labels
    if label not in VALID_LABELS:
        invalid_labels += 1
        continue

    # Check if file exists
    if os.path.isfile(path):

        existing_counter[label] += 1
        total_existing += 1

    else:

        missing_counter[label] += 1
        total_missing += 1

# ─────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────
print("=" * 60)
print("           EMOJI DATASET STATISTICS")
print("=" * 60)

print("\nVALID EXISTING EMOJIS")
print("-" * 30)

for label in VALID_LABELS:
    print(f"{label.capitalize():<10}: {existing_counter[label]}")

print(f"\nTotal Existing Emojis : {total_existing}")

# ─────────────────────────────────────────────
# MISSING FILES REPORT
# ─────────────────────────────────────────────
print("\nMISSING / DELETED EMOJIS")
print("-" * 30)

for label in VALID_LABELS:
    print(f"{label.capitalize():<10}: {missing_counter[label]}")

print(f"\nTotal Missing Emojis  : {total_missing}")

# ─────────────────────────────────────────────
# EXTRA INFO
# ─────────────────────────────────────────────
print("\nOTHER INFORMATION")
print("-" * 30)

print(f"Invalid Labels        : {invalid_labels}")
print(f"MongoDB Records       : {len(documents)}")

# ─────────────────────────────────────────────
# CLASS DISTRIBUTION
# ─────────────────────────────────────────────
if total_existing > 0:

    print("\nCLASS DISTRIBUTION (%)")
    print("-" * 30)

    for label in VALID_LABELS:

        percentage = (
            existing_counter[label] / total_existing
        ) * 100

        print(f"{label.capitalize():<10}: {percentage:.2f}%")

print("\n" + "=" * 60)

# ─────────────────────────────────────────────
# CLOSE CONNECTION
# ─────────────────────────────────────────────
client.close()

print("\n[DONE] Dataset analysis complete.\n")