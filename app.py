from flask import Flask, render_template, request, session
from sentiment import analyze_youtube_comments, _extract_video_id, YOUTUBE_API_KEY
from googleapiclient.discovery import build

app = Flask(__name__)
app.secret_key = "sentiment_analysis_secret_key"


def fetch_video_info(video_id: str) -> dict:
    """Fetch channel name and video title from YouTube API."""
    try:
        youtube  = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        response = youtube.videos().list(part="snippet", id=video_id).execute()
        if response.get("items"):
            snippet = response["items"][0]["snippet"]
            return {
                "channel_name": snippet.get("channelTitle", ""),
                "video_title" : snippet.get("title", ""),
            }
    except Exception:
        pass
    return {"channel_name": "", "video_title": ""}


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    positive = 0
    negative = 0
    neutral  = 0
    comments = []

    overall_positive   = 0.0
    overall_negative   = 0.0
    overall_neutral    = 0.0
    overall_confidence = 0.0
    dominant_sentiment = "N/A"
    channel_name       = ""
    video_title        = ""

    if request.method == 'POST':
        youtube_url   = request.form.get("youtube_url")
        comment_count = int(request.form.get("comment_count", 10))

        video_id = _extract_video_id(youtube_url)
        if video_id:
            info         = fetch_video_info(video_id)
            channel_name = info["channel_name"]
            video_title  = info["video_title"]

        result = analyze_youtube_comments(youtube_url, comment_count)

        if "error" not in result:
            positive           = result["positive"]
            negative           = result["negative"]
            neutral            = result["neutral"]
            comments           = result["comments"]
            overall_positive   = result.get("overall_positive", 0)
            overall_negative   = result.get("overall_negative", 0)
            overall_neutral    = result.get("overall_neutral", 0)
            overall_confidence = result.get("overall_confidence", 0)
            dominant_sentiment = result.get("dominant_sentiment", "N/A")

            session["last_result"]  = result
            session["channel_name"] = channel_name
            session["video_title"]  = video_title

    return render_template(
        "dashboard.html",
        positive=positive,
        negative=negative,
        neutral=neutral,
        comments=comments,
        overall_positive=overall_positive,
        overall_negative=overall_negative,
        overall_neutral=overall_neutral,
        overall_confidence=overall_confidence,
        dominant_sentiment=dominant_sentiment,
        channel_name=channel_name,
        video_title=video_title,
    )


@app.route('/graph', methods=['GET', 'POST'])
def graph():
    positive = 0
    negative = 0
    neutral  = 0
    comments = []

    text_pos  = 0
    text_neg  = 0
    text_neu  = 0

    emoji_pos = 0
    emoji_neg = 0
    emoji_neu = 0

    overall_positive   = 0.0
    overall_negative   = 0.0
    overall_neutral    = 0.0
    overall_confidence = 0.0
    dominant_sentiment = "N/A"
    channel_name       = ""
    video_title        = ""

    if request.method == 'POST':
        youtube_url   = request.form.get("youtube_url")
        comment_count = int(request.form.get("comment_count", 10))
        result        = analyze_youtube_comments(youtube_url, comment_count)
        if "error" not in result:
            session["last_result"] = result
    else:
        result = session.get("last_result", {})

    # Always pull channel info from session (set by dashboard)
    channel_name = session.get("channel_name", "")
    video_title  = session.get("video_title",  "")

    if result and "error" not in result:
        positive           = result["positive"]
        negative           = result["negative"]
        neutral            = result["neutral"]
        comments           = result.get("comments", [])
        overall_positive   = result.get("overall_positive", 0)
        overall_negative   = result.get("overall_negative", 0)
        overall_neutral    = result.get("overall_neutral", 0)
        overall_confidence = result.get("overall_confidence", 0)
        dominant_sentiment = result.get("dominant_sentiment", "N/A")

        total_comments = len(comments)
        if total_comments > 0:
            text_pos  = round(sum(c["text_scores"]["positive"] for c in comments) / total_comments, 2)
            text_neg  = round(sum(c["text_scores"]["negative"] for c in comments) / total_comments, 2)
            text_neu  = round(sum(c["text_scores"]["neutral"]  for c in comments) / total_comments, 2)
            emoji_pos = round(sum(c["emoji_scores"]["positive"] for c in comments) / total_comments, 2)
            emoji_neg = round(sum(c["emoji_scores"]["negative"] for c in comments) / total_comments, 2)
            emoji_neu = round(sum(c["emoji_scores"]["neutral"]  for c in comments) / total_comments, 2)

    total           = positive + negative + neutral
    overall_percent = round((positive / total) * 100) if total > 0 else 0

    return render_template(
        "graph.html",
        positive=positive,
        negative=negative,
        neutral=neutral,
        comments=comments,
        overall_percent=overall_percent,
        text_pos=text_pos,
        text_neg=text_neg,
        text_neu=text_neu,
        emoji_pos=emoji_pos,
        emoji_neg=emoji_neg,
        emoji_neu=emoji_neu,
        overall_positive=overall_positive,
        overall_negative=overall_negative,
        overall_neutral=overall_neutral,
        overall_confidence=overall_confidence,
        dominant_sentiment=dominant_sentiment,
        channel_name=channel_name,
        video_title=video_title,
    )


@app.route('/emoji', methods=['GET', 'POST'])
def emoji():
    positive = 0
    negative = 0
    neutral  = 0

    if request.method == 'POST':
        youtube_url   = request.form.get("youtube_url")
        comment_count = int(request.form.get("comment_count", 10))
        result        = analyze_youtube_comments(youtube_url, comment_count)
        if "error" not in result:
            positive = result["positive"]
            negative = result["negative"]
            neutral  = result["neutral"]

    return render_template("emoji.html", positive=positive, negative=negative, neutral=neutral)


if __name__ == "__main__":
    app.run(debug=True)