from flask import Flask, render_template, request
from sentiment_project.sentiment import analyze_youtube_comments

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    positive = 0
    negative = 0
    neutral = 0
    comments = []
    
    overall_positive = 0.0
    overall_negative = 0.0
    overall_neutral = 0.0
    overall_confidence = 0.0
    dominant_sentiment = "N/A"

    if request.method == 'POST':
        youtube_url = request.form.get("youtube_url")
        comment_count = int(request.form.get("comment_count", 10))

        result = analyze_youtube_comments(
            youtube_url,
            comment_count
        )

        if "error" not in result:
            positive = result["positive"]
            negative = result["negative"]
            neutral = result["neutral"]
            comments = result["comments"]
            
            overall_positive = result.get("overall_positive", 0)
            overall_negative = result.get("overall_negative", 0)
            overall_neutral = result.get("overall_neutral", 0)
            overall_confidence = result.get("overall_confidence", 0)
            dominant_sentiment = result.get("dominant_sentiment", "N/A")

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
        dominant_sentiment=dominant_sentiment
    )


@app.route('/graph', methods=['GET', 'POST'])
def graph():
    positive = 0
    negative = 0
    neutral = 0
    
    text_pos = 0
    text_neg = 0
    text_neu = 0
    
    emoji_pos = 0
    emoji_neg = 0
    emoji_neu = 0
    
    overall_positive = 0.0
    overall_negative = 0.0
    overall_neutral = 0.0
    overall_confidence = 0.0
    dominant_sentiment = "N/A"

    if request.method == 'POST':
        youtube_url = request.form.get("youtube_url")
        comment_count = int(request.form.get("comment_count", 10))

        result = analyze_youtube_comments(
            youtube_url,
            comment_count
        )

        if "error" not in result:
            positive = result["positive"]
            negative = result["negative"]
            neutral = result["neutral"]
            
            overall_positive = result.get("overall_positive", 0)
            overall_negative = result.get("overall_negative", 0)
            overall_neutral = result.get("overall_neutral", 0)
            overall_confidence = result.get("overall_confidence", 0)
            dominant_sentiment = result.get("dominant_sentiment", "N/A")
            
            comments = result.get("comments", [])
            total_comments = len(comments)
            if total_comments > 0:
                text_pos = sum(c["text_scores"]["positive"] for c in comments) / total_comments
                text_neg = sum(c["text_scores"]["negative"] for c in comments) / total_comments
                text_neu = sum(c["text_scores"]["neutral"] for c in comments) / total_comments
                
                emoji_pos = sum(c["emoji_scores"]["positive"] for c in comments) / total_comments
                emoji_neg = sum(c["emoji_scores"]["negative"] for c in comments) / total_comments
                emoji_neu = sum(c["emoji_scores"]["neutral"] for c in comments) / total_comments

    total = positive + negative + neutral

    if total == 0:
        overall_percent = 0
    else:
        overall_percent = round((positive / total) * 100)

    return render_template(
        "graph.html",
        positive=positive,
        negative=negative,
        neutral=neutral,
        overall_percent=overall_percent,
        text_pos=round(text_pos, 2),
        text_neg=round(text_neg, 2),
        text_neu=round(text_neu, 2),
        emoji_pos=round(emoji_pos, 2),
        emoji_neg=round(emoji_neg, 2),
        emoji_neu=round(emoji_neu, 2),
        overall_positive=overall_positive,
        overall_negative=overall_negative,
        overall_neutral=overall_neutral,
        overall_confidence=overall_confidence,
        dominant_sentiment=dominant_sentiment
    )


@app.route('/emoji', methods=['GET', 'POST'])
def emoji():
    positive = 0
    negative = 0
    neutral = 0

    if request.method == 'POST':
        youtube_url = request.form.get("youtube_url")
        comment_count = int(request.form.get("comment_count", 10))

        result = analyze_youtube_comments(
            youtube_url,
            comment_count
        )

        if "error" not in result:
            positive = result["positive"]
            negative = result["negative"]
            neutral = result["neutral"]

    return render_template(
        "emoji.html",
        positive=positive,
        negative=negative,
        neutral=neutral
    )


if __name__ == "__main__":
    app.run(debug=True)