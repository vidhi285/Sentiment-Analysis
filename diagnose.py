"""
diagnose.py
───────────────────────────────────────────────────────
Run this BEFORE retraining to verify:
  1. MongoDB is reachable and has records
  2. Image files actually exist on disk
  3. What the old saved payload looks like
Run: python diagnose.py
───────────────────────────────────────────────────────
this code creates another emoji_path_map.pkl in the same format as before, but with a different structure (dict instead of tuple).
instead of doing manual classification of emoji, it creates multiple emojis for each category (apple, samsung, google, facebook, windows) as same as windows emoji, 
and then maps them to the same label. this way we can have more training data for each category and also avoid the problem of having only one emoji per category which may not be representative enough. 
the new structure of the payload is a dict with keys as category names and values as dicts of emoji paths and labels. for example: 
{
    "apple": {"path1": "label1", "path2": "label1"},
    "samsung": {"path3": "label2", "path4": "label2"}
}
"""

import os
import pickle
from pymongo import MongoClient

MODEL_SAVE_DIR  = r"C:\Users\HP\Documents\Sentiment Analysis Project\saved_models"
EMOJI_MAP_PATH  = os.path.join(MODEL_SAVE_DIR, "emoji_path_map.pkl")
EMOJI_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "emoji_mobilenet.pth")

COLLECTIONS = [
    "apple_emoji", "samsung_emoji", "google_emoji",
    "facebook_emoji", "windows_emoji",
]

print("=" * 60)
print("  EMOJI MODEL DIAGNOSTICS")
print("=" * 60)

# ── 1. Check saved payload ────────────────────────────────────
print("\n[1] Checking saved payload at:", EMOJI_MAP_PATH)
if not os.path.exists(EMOJI_MAP_PATH):
    print("    ✘ File does not exist — needs retraining")
else:
    with open(EMOJI_MAP_PATH, "rb") as f:
        payload = pickle.load(f)

    print(f"    ✔ File exists. Type: {type(payload)}")

    if isinstance(payload, dict):
        print(f"    Keys in payload: {list(payload.keys())}")
        for k, v in payload.items():
            if isinstance(v, dict):
                print(f"      '{k}' → dict with {len(v)} entries")
            elif isinstance(v, list):
                print(f"      '{k}' → list with {len(v)} entries")
            else:
                print(f"      '{k}' → {type(v)}")
    elif isinstance(payload, tuple):
        print(f"    It's a tuple with {len(payload)} elements (old format)")
        for i, item in enumerate(payload):
            print(f"      [{i}] type={type(item)}, len={len(item) if hasattr(item, '__len__') else 'N/A'}")
    else:
        print(f"    Unknown format: {type(payload)}")

# ── 2. Check MongoDB ──────────────────────────────────────────
print("\n[2] Connecting to MongoDB...")
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    client.server_info()
    print("    ✔ MongoDB connected")
    db = client["sentimentDB"]

    total = 0
    for col_name in COLLECTIONS:
        col   = db[col_name]
        count = col.count_documents({})
        print(f"    {col_name}: {count} records")
        total += count

    print(f"    Total records: {total}")

    # ── 3. Sample paths and check disk ───────────────────────
    print("\n[3] Checking if image files exist on disk (sampling 3 per collection)...")
    all_missing = 0
    all_found   = 0

    for col_name in COLLECTIONS:
        col  = db[col_name]
        docs = list(col.find({}, {"_id": 0, "path": 1}).limit(3))
        for doc in docs:
            path = doc.get("path", "")
            exists = os.path.exists(path)
            status = "✔" if exists else "✘"
            if exists:
                all_found += 1
            else:
                all_missing += 1
            print(f"    [{col_name}] {status}  {path}")

    print(f"\n    Found: {all_found}   Missing: {all_missing}")

    if all_missing > 0 and all_found == 0:
        print("\n    ✘ ALL sampled files are missing from disk!")
        print("      Possible reasons:")
        print("      - You moved/renamed the emoji image folder")
        print("      - The paths in MongoDB point to a different machine/drive")
        print("      - The dataset folder was deleted")
        print(f"\n      Expected folder (from sample): {os.path.dirname(docs[0]['path']) if docs else 'N/A'}")
        print("      Please verify this folder exists on your PC.")
    elif all_missing > 0:
        print(f"\n    ⚠ Some files missing. Check if paths are consistent.")
    else:
        print("\n    ✔ All sampled files found on disk — ready to retrain!")

    client.close()

except Exception as e:
    print(f"    ✘ MongoDB error: {e}")
    print("      Make sure MongoDB is running (mongod service)")

# ── 4. Recommendation ─────────────────────────────────────────
print("\n[4] Recommendation:")
if os.path.exists(EMOJI_MODEL_PATH) and os.path.exists(EMOJI_MAP_PATH):
    print("    → Delete both saved files and retrain with the new emoji_model.py:")
    print(f"      del \"{EMOJI_MODEL_PATH}\"")
    print(f"      del \"{EMOJI_MAP_PATH}\"")
    print("      python emoji_model.py")
else:
    print("    → Just run:  python emoji_model.py")

print("\n" + "=" * 60)