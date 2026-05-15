# import pandas as pd
# import emoji
# import re
# import matplotlib.pyplot as plt
# from urllib.parse import urlparse, parse_qs # For extracting video ID from YouTube URL
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer 
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from googleapiclient.discovery import build # For YouTube API access

# file_path = r"C:\Users\HP\Documents\Sentiment Analysis Project\balanced_sentiment_dataset.csv"
# df = pd.read_csv(file_path)
# # We use cleaned text as input and sentiment as output
# df = df.dropna(subset=['clean_text'])
# X = df['clean_text'].astype(str)
# y = df['sentiment']
# #train test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=42
# )
# # TF-IDF Vectorization with unigrams and bigrams, limited to top 5000 features 
# vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000) # cONVERT TEXT TO NUMERIC VECTORS 
# X_train_vec = vectorizer.fit_transform(X_train) # Fit the vectorizer on training data and transform it
# X_test_vec = vectorizer.transform(X_test) # Transform test data using the same vectorizer (don't fit again)

# model = LogisticRegression(max_iter=1000) #logistic regression model for classification
# model.fit(X_train_vec, y_train) 
# accuracy = accuracy_score(y_test, model.predict(X_test_vec))
# print(f"\n Text Model Accuracy: {accuracy*100:.2f}%")


# metadata_path = r"C:\Users\HP\Documents\Sentiment Analysis Project\emoji_metadata.csv"
# emoji_df = pd.read_csv(metadata_path)
# # Create mapping: emojis id → sentiment
# emoji_label_map = {} # We will use the image filename (without extension) as the key, and the label as the value
# for _, row in emoji_df.iterrows():
#     path = row['image_path']
#     label = row['label'] 
#     img_id = path.split("\\")[-1].replace(".png", "") # Extract filename without extension
#     emoji_label_map[img_id] = label # Example: "grinning_face" → "positive"
# print(" Emoji metadata loaded!")


# def get_video_id(url):
#     query = urlparse(url) # Parse the URL to extract components
#     if query.hostname == 'youtu.be': # For short URLs like https://youtu.be/VIDEO_ID
#         return query.path[1:] # The video ID is the path without the leading '/'
#     if query.hostname in ('www.youtube.com', 'youtube.com'): # For standard URLs like https://www.youtube.com/watch?v=VIDEO_ID
#         return parse_qs(query.query)['v'][0] # The video ID is the value of the 'v' parameter in the query string
#     return None


# def extract_emojis(text): # Extract emojis from text using the emoji library's data
#     return [c for c in text if c in emoji.EMOJI_DATA]

# def get_emoji_scores(emojis): # Calculate sentiment scores based on emojis using the emoji_label_mapEmoji metadata generated using Spark is used to map emojis to sentiment.
#     scores = {"positive": 0, "negative": 0, "neutral": 0}
#     if not emojis:
#         return scores
#     for e in emojis: # Convert emoji to its descriptive name (e.g., "😀" → ":grinning_face:") and use that to look up sentiment
#         name = emoji.demojize(e) 
#         idx = str(abs(hash(name)) % 3000) # Generate a pseudo-unique ID for the emoji based on its name (since we don't have actual image files in this context)
#         sentiment = emoji_label_map.get(idx, "neutral") # Default to "neutral" if emoji not found in metadata
#         scores[sentiment] += 1 
#     total = len(emojis) 
#     for k in scores: # Normalize scores to get a distribution (sum to 1)
#         scores[k] /= total # Avoid division by zero since we check for empty emoji list at the start
#     return scores


# def final_decision(text_scores, emoji_scores): # Combine text and emoji scores to make a final sentiment decision. We give priority to text analysis, but emojis can influence the decision in cases of ambiguity (e.g., neutral text with strong emoji sentiment).
#     # TEXT priority
#     text_pos = text_scores["positive"]
#     text_neg = text_scores["negative"]
#     # Neutral condition
#     if 0.45 <= text_pos <= 0.55:
#         if emoji_scores["neutral"] > 0:
#             return "neutral", text_pos
#     # Strong agreement
#     if text_pos > text_neg and emoji_scores["positive"] >= emoji_scores["negative"]:
#         return "positive", text_pos
#     elif text_neg > text_pos and emoji_scores["negative"] >= emoji_scores["positive"]:
#         return "negative", text_neg
#     # fallback (text priority)
#     return ("positive", text_pos) if text_pos > text_neg else ("negative", text_neg)


# api_key = "AIzaSyDkhbQth_t8dWc9nLFnU2g6nYVDQon95Hk"
# youtube = build('youtube', 'v3', developerKey=api_key)
# video_url = input("\n🔗 Enter YouTube Video Link: ")
# video_id = get_video_id(video_url)
# # Make API request to get comments for the specified video ID. We request the "snippet" part which contains the comment text and other details, and we limit to 10 comments for analysis.
# request = youtube.commentThreads().list(
#     part="snippet",
#     videoId=video_id,
#     maxResults=5
# )
# response = request.execute()
# print("\n YouTube Comments Analysis:\n")
# results_count = {"positive": 0, "negative": 0, "neutral": 0} # To keep track of the count of each sentiment category for the final graph

# for i, item in enumerate(response['items'], start=1): # Loop through each comment thread in the API response. Each item contains a top-level comment and possibly replies, but we will focus on the top-level comment for sentiment analysis.
#     comment = item['snippet']['topLevelComment']['snippet']['textDisplay'] # Extract the text of the top-level comment from the API response. The 'textDisplay' field contains the comment text with HTML formatting, but for simplicity we will use it as is. In a more robust implementation, we might want to clean this text to remove HTML tags or decode entities.

#     print("─" * 60)
#     print(f"Comment #{i}")
#     print("Comment is: ", comment)

#     # TEXT ANALYSIS
#     vec = vectorizer.transform([comment]) # Convert the comment text into a vector using the same TF-IDF vectorizer we used for training the model
#     probs = model.predict_proba(vec)[0] # Get the predicted probabilities for each class (negative, positive)
#     text_scores = {
#         "positive": probs[1],
#         "negative": probs[0],
#         "neutral": 0
#     }
#     # EMOJI ANALYSIS
#     emojis = extract_emojis(comment)
#     print(" Emojis:", emojis)
#     emoji_scores = get_emoji_scores(emojis)
#     # FINAL DECISION
#     final_label, confidence = final_decision(text_scores, emoji_scores)
#     results_count[final_label] += 1
#     print(f"\n Final Sentiment: {final_label.upper()}")
#     print(f" Confidence: {confidence*100:.2f}%")


# labels = list(results_count.keys())
# values = list(results_count.values())
# plt.figure()
# plt.bar(labels, values)
# plt.title("YouTube Comment Sentiment Analysis")
# plt.xlabel("Sentiment")
# plt.ylabel("Number of Comments")
# plt.show()
# print("\n DONE")


















# ============================================================
#  main.py
#  Hybrid YouTube Comment Sentiment Analyser
#  Text model  : TF-IDF + Logistic Regression (text_model.py)
#  Emoji model : MobileNetV2                  (emoji_model.py)
#  Weights     : 70% text  +  30% emoji
# ============================================================

import os
import emoji
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pymongo import MongoClient
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build

# ── TEXT MODEL (completely unchanged) ───────────────────────
from text_model import load_text_model, predict_text_sentiment

# ── From emoji_model.py — only reuse constants + build_model ─
# (load_emoji_model / get_emoji_scores_mobilenet are defined
#  below directly in main.py so emoji_model.py needs no change)
from emoji_model import (
    build_model,
    DEVICE,
    LABEL_MAP,
    IMAGE_SIZE,
    NUM_CLASSES,
    MODEL_SAVE_PATH,
    MONGO_URI,
    DB_NAME,
    COLLECTIONS,
)

# ─── CONFIG ─────────────────────────────────────────────────
YOUTUBE_API_KEY = "AIzaSyDkhbQth_t8dWc9nLFnU2g6nYVDQon95Hk"
MAX_COMMENTS    = 10

# Integer index → string label  (reverse of LABEL_MAP)
_IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# Inference transform (no augmentation, same as val in emoji_model.py)
_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ─── EMOJI MODEL LOADER (defined here — no change to emoji_model.py) ─────────
def load_emoji_model():
    """
    Rebuilds MobileNetV2 with the same architecture used in training,
    loads saved weights from emoji_model.pth, and fetches all MongoDB
    records so we can match emoji characters to image files.

    Returns: (model, records)
        model   — MobileNetV2 in eval mode
        records — list of {"path": ..., "label": ...} from all 5 collections
    """
    # Load trained weights
    model = build_model()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    print(f"[Loaded] Emoji model  ← {MODEL_SAVE_PATH}")

    # Fetch all emoji records from MongoDB
    client  = MongoClient(MONGO_URI)
    db      = client[DB_NAME]
    records = []
    for col_name in COLLECTIONS:
        docs = list(db[col_name].find({}, {"path": 1, "label": 1, "_id": 0}))
        records.extend(docs)
    client.close()
    print(f"[Loaded] {len(records)} emoji records from MongoDB")

    return model, records


# ─── RUN MOBILENET ON ONE IMAGE ──────────────────────────────
def _run_mobilenet(image_path, model):
    """Returns {"positive": float, "negative": float, "neutral": float}"""
    image  = Image.open(image_path).convert("RGB")
    tensor = _INFER_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    return {_IDX_TO_LABEL[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}


# ─── EMOJI SCORER (defined here — no change to emoji_model.py) ───────────────
def get_emoji_scores_mobilenet(emojis, model, records):
    """
    Given a list of emoji characters from a comment, looks up each
    emoji's image file in the MongoDB records and runs MobileNet on it.
    Returns averaged scores dict: {"positive", "negative", "neutral"}.
    Falls back to neutral if no emojis or no image match found.
    """
    NEUTRAL = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    if not emojis:
        return NEUTRAL

    # Build filename-stem → normalised path lookup from records
    # Dataset files are named as sequential numbers: 1.png, 2.png …
    path_lookup = {}
    for rec in records:
        raw = rec.get("path", "")
        if not raw:
            continue
        stem = os.path.splitext(os.path.basename(raw))[0]   # "1", "42", …
        path_lookup[stem] = os.path.normpath(raw)

    all_scores = []
    for char in emojis:
        # Try hex codepoint ("1f60a") then decimal ("128522") as filename stem
        for key in [format(ord(char), 'x'), str(ord(char))]:
            matched = path_lookup.get(key)
            if matched and os.path.isfile(matched):
                try:
                    all_scores.append(_run_mobilenet(matched, model))
                except Exception as e:
                    print(f"[Warn] Skipping emoji image {matched}: {e}")
                break   # found a match for this emoji, move to next

    if not all_scores:
        return NEUTRAL

    return {
        label: round(sum(s[label] for s in all_scores) / len(all_scores), 4)
        for label in ["positive", "negative", "neutral"]
    }


# ─── YOUTUBE HELPERS ─────────────────────────────────────────
def get_video_id(url):
    query = urlparse(url)
    if query.hostname == "youtu.be":
        return query.path[1:]
    if query.hostname in ("www.youtube.com", "youtube.com"):
        return parse_qs(query.query).get("v", [None])[0]
    return None


def fetch_youtube_comments(video_id):
    youtube  = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request  = youtube.commentThreads().list(
        part       = "snippet",
        videoId    = video_id,
        maxResults = MAX_COMMENTS,
        textFormat = "plainText",
    )
    response = request.execute()
    return [
        item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        for item in response.get("items", [])
    ]


def extract_emojis_from_text(text):
    return [c for c in text if c in emoji.EMOJI_DATA]


# ─── HYBRID DECISION ─────────────────────────────────────────
def final_decision(text_scores, emoji_scores):
    """70% text + 30% emoji weighted combination."""
    final_scores = {
        "positive": 0.7 * text_scores["positive"]          + 0.3 * emoji_scores["positive"],
        "negative": 0.7 * text_scores["negative"]          + 0.3 * emoji_scores["negative"],
        "neutral" : 0.7 * text_scores.get("neutral", 0.0) + 0.3 * emoji_scores["neutral"],
    }
    label = max(final_scores, key=final_scores.get)
    return label, final_scores[label]


# ─── MAIN ────────────────────────────────────────────────────
def run_analysis():

    # Load text model
    print("\n[Loading] Text model...")
    vectorizer, text_model = load_text_model()

    # Load emoji model (requires emoji_model.pth — run emoji_model.py first)
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"\n[ERROR] {MODEL_SAVE_PATH} not found. Run emoji_model.py first.")
        return

    print("[Loading] Emoji model...")
    emoji_model, records = load_emoji_model()

    # Get YouTube URL
    url      = input("\nEnter YouTube URL: ").strip()
    video_id = get_video_id(url)
    if not video_id:
        print("[ERROR] Invalid YouTube URL.")
        return

    # Fetch comments
    print(f"\n[Info] Fetching up to {MAX_COMMENTS} comments ...")
    comments = fetch_youtube_comments(video_id)
    if not comments:
        print("[Warning] No comments found.")
        return

    # Analyse each comment
    results = {"positive": 0, "negative": 0, "neutral": 0}
    print("\n" + "=" * 60)

    for i, comment in enumerate(comments, 1):
        print(f"\nComment {i}: {comment}")

        text_scores  = predict_text_sentiment(comment, vectorizer, text_model)
        emojis       = extract_emojis_from_text(comment)
        emoji_scores = get_emoji_scores_mobilenet(emojis, emoji_model, records)
        label, conf  = final_decision(text_scores, emoji_scores)
        results[label] += 1

        print(f"  Emojis found : {emojis if emojis else 'none'}")
        print(f"  Text Scores  : {text_scores}")
        print(f"  Emoji Scores : {emoji_scores}")
        print(f"  Final        : {label.upper()}  ({conf * 100:.2f}%)")

    print("\n" + "=" * 60)

    # Summary
    print(f"\n[Summary] {len(comments)} comments analysed:")
    for lbl, cnt in results.items():
        print(f"  {lbl.capitalize():<10}: {cnt}")

    # Bar chart
    colors = ["#4CAF50", "#F44336", "#2196F3"]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(list(results.keys()), list(results.values()),
                   color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, results.values()):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1, str(val),
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.title("Hybrid Sentiment Analysis\n(70% Text + 30% Emoji)", fontsize=13)
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Comments")
    plt.ylim(0, max(results.values()) + 2)
    plt.tight_layout()
    plt.savefig("result.png", dpi=150)
    plt.show()
    print("\nAnalysis Complete  ->  Chart saved as result.png")


if __name__ == "__main__":
    run_analysis()