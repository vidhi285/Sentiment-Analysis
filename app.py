from flask import Flask, render_template, request
from main import analyze_youtube_comments

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

    return render_template(
        "dashboard.html",
        positive=positive,
        negative=negative,
        neutral=neutral,
        comments=comments
    )


@app.route('/graph', methods=['GET', 'POST'])
def graph():
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
        overall_percent=overall_percent
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