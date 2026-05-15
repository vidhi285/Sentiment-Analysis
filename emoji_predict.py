"""
 Loads trained emoji sentiment model and predicts sentiment
 labels for all emoji images inside Apple folder.

 Tasks Performed:
 - Load trained MobileNet/CNN model
 - Read emoji images
 - Resize every image to 48x48
 - Normalize image pixels
 - Predict sentiment label
 - Store prediction results into CSV file

 Labels:
 0 → positive
 1 → negative
 2 → neutral

 Output File:
 apple_emoji_predictions.csv

================================Not necessary============================
"""


import os
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model

# 🔷 LOAD MODEL
model = load_model("emoji_model.h5")

# 🔷 LABEL MAP
reverse_map = {
    0: "positive",
    1: "negative",
    2: "neutral"
}

# 🔷 UNLABELED APPLE PATH
unlabeled_path = r"C:\Users\HP\Documents\Sentiment Analysis Project\dataset\emojis\image\Apple"

# 🔷 PREDICT FUNCTION
def predict_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.reshape(img, (1, 48, 48, 3))

    pred = model.predict(img)
    class_index = np.argmax(pred)
    confidence = np.max(pred)

    return class_index, confidence


# 🔷 LOOP ALL EMOJIS
results = []

for file in os.listdir(unlabeled_path):
    path = os.path.join(unlabeled_path, file)

    result = predict_image(path)
    if result is None:
        continue

    label, confidence = result

    print(f"{file} → {reverse_map[label]} ({confidence:.2f})")

    results.append([file, reverse_map[label], confidence])


# 🔷 SAVE CSV
df = pd.DataFrame(results, columns=["emoji", "predicted_label", "confidence"])
df.to_csv("apple_emoji_predictions.csv", index=False)

print("✅ Predictions saved to apple_emoji_predictions.csv")