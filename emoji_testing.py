"""
════════════════════════════════════════════════════════════════════════════════
  FILE : emoji_testing.py
  ROLE : Prediction app — run this anytime after training is done

  Two modes:
    1. Paste any YouTube comment (with emojis) → predicts sentiment
    2. Type a single emoji using Win+. keyboard → predicts sentiment
    3. Type a full image path manually → predicts sentiment

  Pipeline:
    User input
      ↓
    Extract emoji characters from text
      ↓
    Render each emoji → PNG image in RAM  (no file needed)
      ↓
    Feed to EfficientNet-B0
      ↓
    Show prediction + confidence
════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import io
import re
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

# ── Model save path — must match emoji_model.py ──────────────────────────────
MODEL_PATH = "emoji_model.pth"

# ── Label mapping ─────────────────────────────────────────────────────────────
IDX2LABEL  = {0: "positive", 1: "negative", 2: "neutral"}
LABEL2IDX  = {"positive": 0, "negative": 1, "neutral": 2}

# ── ImageNet normalization ────────────────────────────────────────────────────
IMG_MEAN   = [0.485, 0.456, 0.406]
IMG_STD    = [0.229, 0.224, 0.225]
IMG_SIZE   = 224


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════════════

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}\n")
    return device


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def build_model(device):
    """Rebuild exact same EfficientNet-B0 architecture used in training."""
    model = models.efficientnet_b0(weights=None)

    # Same partial freeze as training (doesn't matter for inference but keeps arch)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, 3),
    )
    return model.to(device)


def load_model(device):
    """Load saved model weights from emoji_model.pth."""
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] No trained model found at: {MODEL_PATH}")
        print("        Run emoji_model.py first to train the model.")
        sys.exit(1)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = build_model(device)

    # Load state dict — handle whatever key format was used
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"  ✅ Model loaded from → {MODEL_PATH}\n")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# EMOJI EXTRACTION FROM TEXT
# ══════════════════════════════════════════════════════════════════════════════

# Unicode ranges that cover all standard emoji blocks
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # geometric shapes extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "\U0001f926-\U0001f937"  # supplemental emoticons
    "\U00010000-\U0010ffff"  # other supplemental
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+",
    flags=re.UNICODE,
)


def extract_emojis(text: str) -> list:
    """
    Extract all emoji characters from a text string.
    Handles single emojis, emoji sequences, and mixed text.

    Example:
        "Amazing video 😊🔥 loved it 😢"
        → ['😊', '🔥', '😢']
    """
    raw_matches = EMOJI_PATTERN.findall(text)

    # Split each match into individual emojis
    individual = []
    for match in raw_matches:
        # Walk through each character
        i = 0
        while i < len(match):
            char = match[i]
            # Check if next char is a variation selector (skip it, keep with emoji)
            if i + 1 < len(match) and match[i + 1] in ('\ufe0f', '\u20e3'):
                individual.append(match[i:i+2])
                i += 2
            else:
                if char not in ('\ufe0f', '\u200d', '\u20e3'):
                    individual.append(char)
                i += 1

    return individual


# ══════════════════════════════════════════════════════════════════════════════
# EMOJI → IMAGE IN RAM
# ══════════════════════════════════════════════════════════════════════════════

def find_emoji_font() -> str | None:
    """
    Find a font file on this Windows machine that can render emoji.
    Returns font path or None if not found.
    """
    # Windows emoji fonts — in order of preference
    candidates = [
        r"C:\Windows\Fonts\seguiemj.ttf",    # Segoe UI Emoji (Windows 10/11)
        r"C:\Windows\Fonts\seguisym.ttf",    # Segoe UI Symbol
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def render_emoji_to_image(emoji_char: str, size: int = 224) -> Image.Image:
    """
    Render an emoji character to a PIL RGB image in memory.
    No file is saved — the image lives purely in RAM.

    Strategy:
    1. Try rendering with Segoe UI Emoji font (Windows built-in)
    2. If font missing, create a simple colored placeholder image

    Returns:
        PIL Image (RGB, size x size)
    """
    # White background canvas
    img  = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_path = find_emoji_font()

    if font_path:
        try:
            from PIL import ImageFont
            # Use a large font so emoji fills the canvas
            font_size = int(size * 0.65)
            font      = ImageFont.truetype(font_path, font_size)

            # Calculate text bounding box for centering
            bbox = draw.textbbox((0, 0), emoji_char, font=font)
            w    = bbox[2] - bbox[0]
            h    = bbox[3] - bbox[1]
            x    = (size - w) // 2 - bbox[0]
            y    = (size - h) // 2 - bbox[1]

            draw.text((x, y), emoji_char, font=font, embedded_color=True)
            return img

        except Exception:
            pass  # Fall through to placeholder

    # ── Fallback: colored rectangle with unicode codepoint ────────────────────
    # This happens only if Segoe UI Emoji is missing (very rare on Windows)
    codepoint = ord(emoji_char[0]) if emoji_char else 0
    r = (codepoint * 37) % 200 + 55
    g = (codepoint * 53) % 200 + 55
    b = (codepoint * 71) % 200 + 55
    img = Image.new("RGB", (size, size), color=(r, g, b))
    return img


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])


def predict_from_image(model, img: Image.Image, device) -> dict:
    """
    Run EfficientNet-B0 inference on a PIL Image.

    Returns:
        dict with predicted_label, confidence, probabilities
    """
    tf     = get_val_transform()
    tensor = tf(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100

    return {
        "label"      : IDX2LABEL[pred_idx],
        "confidence" : confidence,
        "probs"      : {IDX2LABEL[i]: float(probs[i]) * 100 for i in range(3)},
    }


def predict_emoji_char(model, emoji_char: str, device) -> dict:
    """Render emoji character to image, then predict."""
    img = render_emoji_to_image(emoji_char, size=IMG_SIZE)
    return predict_from_image(model, img, device)


def predict_image_path(model, path: str, device) -> dict:
    """Load image from disk path, then predict."""
    img = Image.open(path).convert("RGB")
    return predict_from_image(model, img, device)


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

SENTIMENT_ICON = {"positive": "😊", "negative": "😞", "neutral": "😐"}
SENTIMENT_COLOR = {"positive": "green", "negative": "red", "neutral": "gray"}


def display_single_result(emoji_char: str, result: dict):
    """Print prediction result for one emoji."""
    label = result["label"]
    conf  = result["confidence"]
    icon  = SENTIMENT_ICON.get(label, "")

    print(f"\n  ┌{'─'*45}┐")
    print(f"  │  Emoji      : {emoji_char:<30}│")
    print(f"  │  Sentiment  : {icon} {label.upper():<27}│")
    print(f"  │  Confidence : {conf:>6.2f}%{'':<28}│")
    print(f"  ├{'─'*45}┤")
    print(f"  │  Class Probabilities:{'':24}│")
    for cls, prob in sorted(result["probs"].items(), key=lambda x: -x[1]):
        bar   = "▓" * int(prob / 5)
        arrow = " ◀" if cls == label else ""
        print(f"  │    {cls:<10}: {prob:>6.2f}%  {bar:<10}{arrow:<2}{'':>3}│")
    print(f"  └{'─'*45}┘\n")


def display_comment_result(comment: str, results: list):
    """
    Print overall sentiment for a full comment with multiple emojis.

    Results is a list of (emoji_char, prediction_dict).
    Overall sentiment = majority vote across all emojis.
    """
    if not results:
        print("\n  [INFO] No emojis found in this comment.\n")
        return

    print(f"\n{'═'*55}")
    print(f"  COMMENT ANALYSIS")
    print(f"{'─'*55}")
    print(f"  Comment : {comment[:60]}{'...' if len(comment)>60 else ''}")
    print(f"{'─'*55}")
    print(f"  Emojis found : {len(results)}\n")

    # Per-emoji results
    label_votes = []
    for emoji_char, result in results:
        label = result["label"]
        conf  = result["confidence"]
        icon  = SENTIMENT_ICON.get(label, "")
        label_votes.append(label)
        print(f"    {emoji_char}  →  {icon} {label.upper():<10}  ({conf:.1f}%)")

    # Overall sentiment = majority vote
    from collections import Counter
    vote_counts  = Counter(label_votes)
    overall      = vote_counts.most_common(1)[0][0]
    overall_icon = SENTIMENT_ICON.get(overall, "")

    print(f"\n{'─'*55}")
    print(f"  OVERALL SENTIMENT : {overall_icon}  {overall.upper()}")
    print(f"  Vote breakdown    : ", end="")
    for label, count in vote_counts.most_common():
        print(f"{label}={count}", end="  ")
    print(f"\n{'═'*55}\n")

    # Optional: show image popup
    _show_comment_image(comment, results, overall)


def _show_comment_image(comment, results, overall):
    """Optional matplotlib popup showing emojis + overall result."""
    try:
        import matplotlib.pyplot as plt

        n    = len(results)
        cols = min(n, 5)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
        fig.suptitle(
            f"Overall: {overall.upper()}  {SENTIMENT_ICON.get(overall,'')}",
            fontsize=14, fontweight="bold",
            color=SENTIMENT_COLOR.get(overall, "black"),
        )

        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for i, (emoji_char, result) in enumerate(results):
            img = render_emoji_to_image(emoji_char, size=224)
            axes_flat[i].imshow(img)
            axes_flat[i].axis("off")
            label = result["label"]
            conf  = result["confidence"]
            axes_flat[i].set_title(
                f"{emoji_char}\n{label} ({conf:.0f}%)",
                fontsize=9,
                color=SENTIMENT_COLOR.get(label, "black"),
            )

        # Hide unused axes
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        plt.tight_layout()
        plt.show()

    except ImportError:
        pass  # matplotlib not required


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE PREDICTION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_predictor(model, device):
    """
    Main interactive loop. Three input modes:

    MODE A — Full comment (text with emojis):
        Paste: "This video is amazing 😊🔥 love it!"
        → extracts all emojis → predicts each → shows overall sentiment

    MODE B — Single emoji (Win+. keyboard):
        Press Win+. → click emoji → press Enter
        → instant single prediction

    MODE C — Image file path (manual):
        Type: C:\\Users\\HP\\...\\1063.png
        → loads image → predicts

    Type 'quit' to exit.
    """
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║       🎯  EMOJI SENTIMENT PREDICTOR                 ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║                                                      ║")
    print("║  MODE A — Paste a YouTube comment with emojis:      ║")
    print("║    > Amazing video 😊🔥 loved it 😢                  ║")
    print("║                                                      ║")
    print("║  MODE B — Type a single emoji (press Win + .):      ║")
    print("║    > 😊                                              ║")
    print("║                                                      ║")
    print("║  MODE C — Type a full image path:                   ║")
    print("║    > C:\\Users\\HP\\...\\Windows\\1063.png              ║")
    print("║                                                      ║")
    print("║  Type  quit  to exit                                ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    while True:
        try:
            user_input = input("  Input > ").strip().strip('"').strip("'")
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("  Goodbye!\n")
            break

        # ── Detect input mode ─────────────────────────────────────────────────

        looks_like_path = (
            os.sep in user_input
            or "/" in user_input
            or "\\" in user_input
            or user_input.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        )

        emojis_in_input = extract_emojis(user_input)

        # ── MODE C: File path ─────────────────────────────────────────────────
        if looks_like_path:
            img_path = str(Path(user_input))
            if not os.path.isfile(img_path):
                print(f"\n  [ERROR] File not found: {img_path}\n")
                continue
            try:
                result = predict_image_path(model, img_path, device)
                display_single_result(os.path.basename(img_path), result)
            except Exception as e:
                print(f"\n  [ERROR] {e}\n")

        # ── MODE A: Comment with multiple emojis ──────────────────────────────
        elif len(emojis_in_input) > 1 or (len(emojis_in_input) == 1 and len(user_input) > 4):
            print(f"\n  Analyzing comment...")
            results = []
            for emoji_char in emojis_in_input:
                try:
                    result = predict_emoji_char(model, emoji_char, device)
                    results.append((emoji_char, result))
                except Exception as e:
                    print(f"  [WARN] Could not process emoji {emoji_char}: {e}")
            display_comment_result(user_input, results)

        # ── MODE B: Single emoji ──────────────────────────────────────────────
        elif len(emojis_in_input) == 1:
            emoji_char = emojis_in_input[0]
            try:
                result = predict_emoji_char(model, emoji_char, device)
                display_single_result(emoji_char, result)
            except Exception as e:
                print(f"\n  [ERROR] {e}\n")

        # ── No emoji found ────────────────────────────────────────────────────
        else:
            print("\n  [INFO] No emoji detected in your input.")
            print("         This model works on emoji images.")
            print("         Try: press Win+.  and click an emoji, then Enter.\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def print_banner():
    print("\n" + "═"*55)
    print("   EMOJI SENTIMENT PREDICTOR")
    print("   EfficientNet-B0  |  PyTorch")
    print("   Positive · Negative · Neutral")
    print("═"*55 + "\n")


def main():
    print_banner()
    device = get_device()
    model  = load_model(device)
    run_predictor(model, device)


if __name__ == "__main__":
    main()