from googleapiclient.discovery import build
import re

# -----------------------------
# STEP 1: ADD YOUR YOUTUBE API KEY
# -----------------------------
API_KEY = "AIzaSyDzI3h2GdWBTm58NVEYylhpexjKF4pnZBQ"


# -----------------------------
# STEP 2: EXTRACT VIDEO ID FROM URL
# -----------------------------
def extract_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return None


# -----------------------------
# STEP 3: FETCH COMMENTS FROM YOUTUBE
# -----------------------------
def get_youtube_comments(video_id, max_comments=20):
    youtube = build(
        "youtube",
        "v3",
        developerKey=API_KEY
    )

    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_comments,
        textFormat="plainText"
    )

    response = request.execute()

    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments


# -----------------------------
# STEP 4: SIMPLE SENTIMENT PREDICTION
# (Replace later with your ML model)
# -----------------------------
def predict_sentiment(comment):
    comment = comment.lower()

    positive_words = [
        "good", "great", "awesome", "amazing",
        "love", "excellent", "nice", "best",
        "helpful", "super", "beautiful"
    ]

    negative_words = [
        "bad", "worst", "hate", "poor",
        "useless", "boring", "angry",
        "waste", "terrible"
    ]

    for word in positive_words:
        if word in comment:
            return "Positive"

    for word in negative_words:
        if word in comment:
            return "Negative"

    return "Neutral"


# -----------------------------
# STEP 5: FINAL ANALYSIS
# -----------------------------
def analyze_youtube_comments(url, max_comments=20):
    video_id = extract_video_id(url)

    if not video_id:
        return {
            "error": "Invalid YouTube URL"
        }

    comments = get_youtube_comments(video_id, max_comments)

    positive = 0
    negative = 0
    neutral = 0

    analyzed_comments = []

    for comment in comments:
        sentiment = predict_sentiment(comment)

        if sentiment == "Positive":
            positive += 1
        elif sentiment == "Negative":
            negative += 1
        else:
            neutral += 1

        analyzed_comments.append({
            "comment": comment,
            "sentiment": sentiment
        })

    return {
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "comments": analyzed_comments
    }


# -----------------------------
# STEP 6: TEST RUN
# -----------------------------
if __name__ == "__main__":
    youtube_url = input("Enter YouTube URL: ")

    result = analyze_youtube_comments(
        youtube_url,
        max_comments=20
    )

    if "error" in result:
        print(result["error"])

    else:
        print("\n--- FINAL RESULT ---\n")

        print("Positive:", result["positive"])
        print("Negative:", result["negative"])
        print("Neutral :", result["neutral"])

        print("\n--- COMMENTS ANALYSIS ---\n")

        for item in result["comments"]:
            print(
                item["comment"],
                "→",
                item["sentiment"]
            )