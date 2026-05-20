# ============================================================
#  sentiment.py
#  Backend AI engine for Flask app (app.py)
#  Text model  : SBERT + Keras Deep Learning  (sbert_keras_model.h5)
#  Emoji model : EfficientNet-B0              (emoji_model.pth)
#  Weights     : 75% text  +  25% emoji
# ============================================================

import os
import re
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Sequential       # type: ignore
from tensorflow.keras.layers import Dense, Dropout  # type: ignore
from sentence_transformers import SentenceTransformer

# ════════════════════════════════════════════════════════════
#  CONFIG — change filenames here if needed
# ════════════════════════════════════════════════════════════
TEXT_MODEL_PATH  = "sbert_keras_model.h5"
EMOJI_MODEL_PATH = "emoji_model.pth"
YOUTUBE_API_KEY  = "AIzaSyDkhbQth_t8dWc9nLFnU2g6nYVDQon95Hk"
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

TEXT_WEIGHT  = 0.75
EMOJI_WEIGHT = 0.25

IDX2LABEL = {0: "positive", 1: "negative", 2: "neutral"}
IMG_SIZE  = 224
IMG_MEAN  = [0.485, 0.456, 0.406]
IMG_STD   = [0.229, 0.224, 0.225]
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
    "]+",
    flags=re.UNICODE,
)


# ════════════════════════════════════════════════════════════
#  EMOJI UTILITIES
# ════════════════════════════════════════════════════════════

def extract_emojis(text: str) -> list:
    raw_matches = EMOJI_PATTERN.findall(text)
    individual  = []
    for match in raw_matches:
        i = 0
        while i < len(match):
            char = match[i]
            if i + 1 < len(match) and match[i + 1] in ('\ufe0f', '\u20e3'):
                individual.append(match[i:i+2])
                i += 2
            else:
                if char not in ('\ufe0f', '\u200d', '\u20e3'):
                    individual.append(char)
                i += 1
    return individual


def _find_emoji_font():
    candidates = [
        r"C:\Windows\Fonts\seguiemj.ttf",
        r"C:\Windows\Fonts\seguisym.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def render_emoji_to_image(emoji_char: str, size: int = 224) -> Image.Image:
    """Render emoji character → PIL RGB image in RAM (no file saved)."""
    img  = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_path = _find_emoji_font()
    if font_path:
        try:
            font_size = int(size * 0.65)
            font      = ImageFont.truetype(font_path, font_size)
            bbox      = draw.textbbox((0, 0), emoji_char, font=font)
            w, h      = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x         = (size - w) // 2 - bbox[0]
            y         = (size - h) // 2 - bbox[1]
            draw.text((x, y), emoji_char, font=font, embedded_color=True)
            return img
        except Exception:
            pass
    cp = ord(emoji_char[0]) if emoji_char else 0
    return Image.new("RGB", (size, size),
                     color=((cp * 37) % 200 + 55,
                            (cp * 53) % 200 + 55,
                            (cp * 71) % 200 + 55))


# ════════════════════════════════════════════════════════════
#  EMOJI MODEL  (EfficientNet-B0)
# ════════════════════════════════════════════════════════════

def _build_efficientnet() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, 3),
    )
    return model.to(DEVICE)


_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])


def _load_emoji_model() -> nn.Module:
    if not os.path.isfile(EMOJI_MODEL_PATH):
        raise FileNotFoundError(f"Emoji model not found: {EMOJI_MODEL_PATH}")
    checkpoint = torch.load(EMOJI_MODEL_PATH, map_location=DEVICE)
    model = _build_efficientnet()
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"[✓] Emoji model loaded ← {EMOJI_MODEL_PATH}")
    return model


def _predict_emoji_image(img: Image.Image, emoji_model: nn.Module) -> dict:
    tensor = _INFER_TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(emoji_model(tensor), dim=1)[0].cpu().numpy()
    return {IDX2LABEL[i]: float(probs[i]) for i in range(3)}


def get_emoji_scores(emojis: list, emoji_model: nn.Module) -> dict:
    NEUTRAL = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
    if not emojis:
        return NEUTRAL
    all_scores = []
    for char in emojis:
        try:
            img = render_emoji_to_image(char, size=IMG_SIZE)
            all_scores.append(_predict_emoji_image(img, emoji_model))
        except Exception:
            pass
    if not all_scores:
        return NEUTRAL
    return {
        label: round(sum(s[label] for s in all_scores) / len(all_scores), 4)
        for label in ["positive", "negative", "neutral"]
    }


# ════════════════════════════════════════════════════════════
#  TEXT MODEL  (SBERT + Keras)
# ════════════════════════════════════════════════════════════

def _load_text_model():
    if not os.path.isfile(TEXT_MODEL_PATH):
        raise FileNotFoundError(f"Text model not found: {TEXT_MODEL_PATH}")

    print(f"[~] Loading SBERT encoder ({SBERT_MODEL_NAME}) ...")
    sbert = SentenceTransformer(SBERT_MODEL_NAME)

    print(f"[~] Loading Keras model ← {TEXT_MODEL_PATH} ...")

    # Rebuild architecture manually to avoid Keras version mismatch
    # (quantization_config issue when loading with keras_load_model)
    keras_model = Sequential([
        Dense(256, activation='relu', input_dim=384),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    keras_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    keras_model.load_weights(TEXT_MODEL_PATH)

    print(f"[✓] Text model loaded ← {TEXT_MODEL_PATH}")
    return sbert, keras_model


def predict_text_sentiment(comment: str, sbert, keras_model) -> dict:
    """
    SBERT encodes text → Keras predicts → returns 3-class scores.
    Keras outputs single sigmoid (0=negative, 1=positive).
    """
    emb  = sbert.encode([comment])
    prob = float(keras_model.predict(emb, verbose=0)[0][0])
    return {
        "positive": prob,
        "negative": 1.0 - prob,
        "neutral" : 0.0,
    }


# ════════════════════════════════════════════════════════════
#  HYBRID FUSION  (75% text + 25% emoji)
# ════════════════════════════════════════════════════════════

def _final_decision(text_scores: dict, emoji_scores: dict) -> tuple:
    fused = {
        label: TEXT_WEIGHT  * text_scores.get(label, 0.0)
               + EMOJI_WEIGHT * emoji_scores.get(label, 0.0)
        for label in ["positive", "negative", "neutral"]
    }
    label = max(fused, key=fused.get)
    return label, fused[label], fused


# ════════════════════════════════════════════════════════════
#  MODEL LOADING — once at Flask startup, cached globally
# ════════════════════════════════════════════════════════════

print("[Loading AI models — please wait...]")
_SBERT, _KERAS_MODEL = _load_text_model()
_EMOJI_MODEL         = _load_emoji_model()
print("[All models ready]\n")


# ════════════════════════════════════════════════════════════
#  YOUTUBE HELPERS
# ════════════════════════════════════════════════════════════

def _extract_video_id(url: str):
    query = urlparse(url)
    if query.hostname == "youtu.be":
        return query.path[1:]
    if query.hostname in ("www.youtube.com", "youtube.com"):
        return parse_qs(query.query).get("v", [None])[0]
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def _fetch_comments(video_id: str, max_results: int) -> list:
    youtube         = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments        = []
    next_page_token = None
    while len(comments) < max_results:
        fetch_count = min(100, max_results - len(comments))
        request = youtube.commentThreads().list(
            part       = "snippet",
            videoId    = video_id,
            maxResults = fetch_count,
            textFormat = "plainText",
            pageToken  = next_page_token,
        )
        response        = request.execute()
        next_page_token = response.get("nextPageToken")
        for item in response.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(text)
        if not next_page_token:
            break
    return comments[:max_results]


# ════════════════════════════════════════════════════════════
#  MAIN PUBLIC FUNCTION  (called by app.py)
# ════════════════════════════════════════════════════════════

def analyze_youtube_comments(youtube_url: str, max_comments: int = 10) -> dict:
    """
    Full pipeline: fetch YouTube comments → analyse with both models
    → return structured result dict expected by app.py / HTML templates.
    """
    # Validate URL
    video_id = _extract_video_id(youtube_url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}

    # Fetch comments
    try:
        raw_comments = _fetch_comments(video_id, max_comments)
    except Exception as e:
        return {"error": f"YouTube API error: {str(e)}"}

    if not raw_comments:
        return {"error": "No comments found for this video"}

    # Analyse each comment
    tally = {"positive": 0, "negative": 0, "neutral": 0}

    total_fused_pos = 0.0
    total_fused_neg = 0.0
    total_fused_neu = 0.0
    total_conf      = 0.0

    analyzed_comments = []

    for comment in raw_comments:

        # Text model: SBERT + Keras
        text_scores  = predict_text_sentiment(comment, _SBERT, _KERAS_MODEL)

        # Emoji model: extract → render → EfficientNet
        emojis       = extract_emojis(comment)
        emoji_scores = get_emoji_scores(emojis, _EMOJI_MODEL)

        # Hybrid fusion
        label, conf, fused = _final_decision(text_scores, emoji_scores)
        tally[label] += 1

        # Accumulate for overall averages (scale to 0-100)
        total_fused_pos += fused["positive"] * 100
        total_fused_neg += fused["negative"] * 100
        total_fused_neu += fused["neutral"]  * 100
        total_conf      += conf * 100

        analyzed_comments.append({
            "comment"        : comment,
            "cleaned_text"   : comment.strip(),
            "emojis"         : emojis,
            "text_scores"    : {
                "positive": round(text_scores["positive"] * 100, 2),
                "negative": round(text_scores["negative"] * 100, 2),
                "neutral" : round(text_scores["neutral"]  * 100, 2),
            },
            "emoji_scores"   : {
                "positive": round(emoji_scores["positive"] * 100, 2),
                "negative": round(emoji_scores["negative"] * 100, 2),
                "neutral" : round(emoji_scores["neutral"]  * 100, 2),
            },
            "final_sentiment": label.upper(),
            "confidence"     : round(conf * 100, 2),
        })

    # Overall averages
    n = len(analyzed_comments)

    overall_pos  = round(total_fused_pos / n, 2)
    overall_neg  = round(total_fused_neg / n, 2)
    overall_neu  = round(total_fused_neu / n, 2)
    overall_conf = round(total_conf / n, 2)

    scores_map = {
        "POSITIVE": overall_pos,
        "NEGATIVE": overall_neg,
        "NEUTRAL" : overall_neu,
    }
    dominant = max(scores_map, key=scores_map.get)

    return {
        "positive"          : tally["positive"],
        "negative"          : tally["negative"],
        "neutral"           : tally["neutral"],
        "overall_positive"  : overall_pos,
        "overall_negative"  : overall_neg,
        "overall_neutral"   : overall_neu,
        "overall_confidence": overall_conf,
        "dominant_sentiment": dominant,
        "comments"          : analyzed_comments,
    }