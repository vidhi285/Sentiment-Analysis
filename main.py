# ============================================================
#  main.py
#  Hybrid YouTube Comment Sentiment Analyser
#  Text model  : SBERT + Keras Deep Learning  (sbert_keras_model.h5)
#  Emoji model : EfficientNet-B0              (emoji_model.pth)
#  Weights     : 75% text  +  25% emoji
# ============================================================

import os
import sys
import re
import warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import Counter
from urllib.parse import urlparse, parse_qs
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
from googleapiclient.discovery import build

# ── Suppress noisy warnings ──────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── TensorFlow / Keras (text model) ─────────────────────────
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model  # type: ignore
from sentence_transformers import SentenceTransformer

# ════════════════════════════════════════════════════════════
#  ▶  USER CONFIGURATION — edit only these two lines
# ════════════════════════════════════════════════════════════
TEXT_MODEL_PATH  = "sbert_keras_model.h5"   # ← replace with your friend's filename if different
EMOJI_MODEL_PATH = "emoji_model.pth"
# ════════════════════════════════════════════════════════════

YOUTUBE_API_KEY  = "AIzaSyDkhbQth_t8dWc9nLFnU2g6nYVDQon95Hk"
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

TEXT_WEIGHT  = 0.75
EMOJI_WEIGHT = 0.25

LABEL2IDX = {"positive": 0, "negative": 1, "neutral": 2}
IDX2LABEL  = {0: "positive", 1: "negative", 2: "neutral"}

IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════
#  EMOJI UTILITIES
# ════════════════════════════════════════════════════════════

def extract_emojis(text: str) -> list:
    """Extract individual emoji characters from a text string."""
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


def _find_emoji_font() -> str | None:
    """Find a Windows emoji-capable font file."""
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
    """
    Render an emoji character to a PIL RGB image in memory (no file saved).
    Uses Segoe UI Emoji font on Windows; falls back to a coloured placeholder.
    """
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

    # Fallback: coloured rectangle
    cp  = ord(emoji_char[0]) if emoji_char else 0
    img = Image.new("RGB", (size, size),
                    color=((cp * 37) % 200 + 55,
                           (cp * 53) % 200 + 55,
                           (cp * 71) % 200 + 55))
    return img


# ════════════════════════════════════════════════════════════
#  EMOJI MODEL  (EfficientNet-B0)
# ════════════════════════════════════════════════════════════

def _build_efficientnet() -> nn.Module:
    """Rebuild exact EfficientNet-B0 architecture used during training."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, 3),
    )
    return model.to(DEVICE)


def load_emoji_model() -> nn.Module:
    """Load trained EfficientNet-B0 weights from emoji_model.pth."""
    if not os.path.isfile(EMOJI_MODEL_PATH):
        print(f"[ERROR] Emoji model not found: {EMOJI_MODEL_PATH}")
        sys.exit(1)

    checkpoint = torch.load(EMOJI_MODEL_PATH, map_location=DEVICE)
    model = _build_efficientnet()

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"  [✓] Emoji model loaded  ← {EMOJI_MODEL_PATH}")
    return model


_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])


def predict_emoji_image(img: Image.Image, emoji_model: nn.Module) -> dict:
    """
    Run EfficientNet inference on a PIL Image.
    Returns {"positive": float, "negative": float, "neutral": float}
    """
    tensor = _INFER_TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(emoji_model(tensor), dim=1)[0].cpu().numpy()
    return {IDX2LABEL[i]: float(probs[i]) for i in range(3)}


def get_emoji_scores(emojis: list, emoji_model: nn.Module) -> dict:
    """
    For each emoji in the list:
      1. Render emoji → PIL image in RAM
      2. Run EfficientNet → probabilities
    Return averaged scores. Falls back to neutral if no emojis.
    """
    NEUTRAL = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    if not emojis:
        return NEUTRAL

    all_scores = []
    for char in emojis:
        try:
            img    = render_emoji_to_image(char, size=IMG_SIZE)
            scores = predict_emoji_image(img, emoji_model)
            all_scores.append(scores)
        except Exception as e:
            print(f"    [Warn] Could not process emoji '{char}': {e}")

    if not all_scores:
        return NEUTRAL

    return {
        label: round(sum(s[label] for s in all_scores) / len(all_scores), 4)
        for label in ["positive", "negative", "neutral"]
    }


# ════════════════════════════════════════════════════════════
#  TEXT MODEL  (SBERT + Keras)
# ════════════════════════════════════════════════════════════

def load_text_model():
    if not os.path.isfile(TEXT_MODEL_PATH):
        print(f"[ERROR] Text model not found: {TEXT_MODEL_PATH}")
        sys.exit(1)

    print(f"  [~] Loading SBERT encoder ({SBERT_MODEL_NAME}) ...")
    sbert = SentenceTransformer(SBERT_MODEL_NAME)

    print(f"  [~] Loading Keras model    ← {TEXT_MODEL_PATH} ...")
    import h5py
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    model = Sequential([
        Dense(256, activation='relu', input_dim=384),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights(TEXT_MODEL_PATH)

    print(f"  [✓] Text model loaded")
    return sbert, model


def predict_text_sentiment(comment: str,
                           sbert: SentenceTransformer,
                           keras_model) -> dict:
    """
    Embed comment with SBERT → feed to Keras → return 3-class scores.

    The Keras model outputs a single sigmoid value (binary: 0=neg, 1=pos).
    We convert it to a 3-class distribution:
      positive = prob
      negative = 1 - prob
      neutral  = 0   (binary model has no neutral; hybrid fusion handles it)
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

def final_decision(text_scores: dict, emoji_scores: dict) -> tuple:
    """
    Weighted combination: 75% text + 25% emoji.
    Returns (label, confidence_float).
    """
    fused = {
        label: TEXT_WEIGHT  * text_scores.get(label, 0.0)
               + EMOJI_WEIGHT * emoji_scores.get(label, 0.0)
        for label in ["positive", "negative", "neutral"]
    }
    label = max(fused, key=fused.get)
    return label, fused[label]


# ════════════════════════════════════════════════════════════
#  YOUTUBE HELPERS
# ════════════════════════════════════════════════════════════

def get_video_id(url: str) -> str | None:
    query = urlparse(url)
    if query.hostname == "youtu.be":
        return query.path[1:]
    if query.hostname in ("www.youtube.com", "youtube.com"):
        return parse_qs(query.query).get("v", [None])[0]
    return None


def fetch_youtube_comments(video_id: str, max_results: int) -> list:
    """Fetch top-level comments from a YouTube video."""
    youtube  = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments = []
    next_page_token = None

    while len(comments) < max_results:
        fetch_count = min(100, max_results - len(comments))
        request = youtube.commentThreads().list(
            part           = "snippet",
            videoId        = video_id,
            maxResults     = fetch_count,
            textFormat     = "plainText",
            pageToken      = next_page_token,
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
#  DISPLAY HELPERS
# ════════════════════════════════════════════════════════════

def _bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def print_comment_result(i: int, comment: str,
                         text_scores: dict, emoji_scores: dict,
                         emojis: list, label: str, conf: float):
    icons = {"positive": "🟢", "negative": "🔴", "neutral": "🔵"}
    print(f"\n  {'─'*60}")
    print(f"  Comment #{i}")
    snippet = comment[:80] + ("..." if len(comment) > 80 else "")
    print(f"  Text    : {snippet}")
    print(f"  Emojis  : {' '.join(emojis) if emojis else '(none)'}")
    print(f"\n  Text Scores  (75% weight):")
    for lbl in ["positive", "negative", "neutral"]:
        print(f"    {lbl:<10}: {text_scores[lbl]*100:>6.2f}%  {_bar(text_scores[lbl])}")
    print(f"\n  Emoji Scores (25% weight):")
    for lbl in ["positive", "negative", "neutral"]:
        print(f"    {lbl:<10}: {emoji_scores[lbl]*100:>6.2f}%  {_bar(emoji_scores[lbl])}")
    print(f"\n  {icons.get(label,'⚪')}  Final Sentiment : {label.upper()}   "
          f"(confidence: {conf*100:.2f}%)")


# ════════════════════════════════════════════════════════════
#  CHART
# ════════════════════════════════════════════════════════════

def show_chart(results: dict, total: int):
    colors = {"positive": "#4CAF50", "negative": "#F44336", "neutral": "#2196F3"}
    labels = list(results.keys())
    values = list(results.values())
    cols   = [colors[l] for l in labels]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=cols, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        pct = f"{val/total*100:.1f}%" if total else "0%"
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.15,
                 f"{val}  ({pct})",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.title(
        f"YouTube Comment Sentiment Analysis\n"
        f"(75% SBERT Text + 25% EfficientNet Emoji)  —  {total} comments",
        fontsize=12,
    )
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Comments")
    plt.ylim(0, max(values, default=1) + 2)
    plt.tight_layout()
    plt.savefig("result.png", dpi=150)
    print("\n  Chart saved → result.png")
    plt.show()


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def print_banner():
    print("\n" + "═" * 60)
    print("   HYBRID YOUTUBE COMMENT SENTIMENT ANALYSER")
    print("   Text  : SBERT + Keras       (75% weight)")
    print("   Emoji : EfficientNet-B0     (25% weight)")
    print("   Labels: Positive | Negative | Neutral")
    print("═" * 60)


def run_analysis():
    print_banner()

    # ── Load models ────────────────────────────────────────
    print("\n[Loading models...]")
    sbert, keras_model = load_text_model()
    emoji_model        = load_emoji_model()
    print("[All models loaded]\n")

    # ── YouTube URL ─────────────────────────────────────────
    url      = input("🔗 Enter YouTube Video URL : ").strip()
    video_id = get_video_id(url)
    if not video_id:
        print("[ERROR] Invalid YouTube URL. Exiting.")
        return

    # ── How many comments ───────────────────────────────────
    while True:
        try:
            n = int(input("💬 How many comments to fetch? : ").strip())
            if n < 1:
                raise ValueError
            break
        except ValueError:
            print("  Please enter a positive integer.")

    # ── Fetch ────────────────────────────────────────────────
    print(f"\n[Fetching up to {n} comments from YouTube...]\n")
    try:
        comments = fetch_youtube_comments(video_id, n)
    except Exception as e:
        print(f"[ERROR] YouTube API error: {e}")
        return

    if not comments:
        print("[Warning] No comments found for this video.")
        return

    print(f"[Fetched {len(comments)} comment(s)]\n")
    print("=" * 60)

    # ── Analyse ──────────────────────────────────────────────
    tally = {"positive": 0, "negative": 0, "neutral": 0}

    for i, comment in enumerate(comments, 1):

        # Text prediction
        text_scores  = predict_text_sentiment(comment, sbert, keras_model)

        # Emoji prediction (render emoji → image → EfficientNet)
        emojis       = extract_emojis(comment)
        emoji_scores = get_emoji_scores(emojis, emoji_model)

        # Hybrid fusion
        label, conf  = final_decision(text_scores, emoji_scores)
        tally[label] += 1

        print_comment_result(i, comment, text_scores, emoji_scores,
                             emojis, label, conf)

    # ── Summary ──────────────────────────────────────────────
    total = len(comments)
    print("\n" + "=" * 60)
    print(f"\n  SUMMARY — {total} comment(s) analysed")
    print(f"  {'─'*30}")
    icons = {"positive": "🟢", "negative": "🔴", "neutral": "🔵"}
    for lbl, cnt in tally.items():
        pct = f"{cnt/total*100:.1f}%" if total else "0%"
        print(f"  {icons[lbl]} {lbl.capitalize():<12}: {cnt:>3}  ({pct})")

    # ── Chart ────────────────────────────────────────────────
    show_chart(tally, total)
    print("\n  Analysis complete!\n")


if __name__ == "__main__":
    run_analysis()