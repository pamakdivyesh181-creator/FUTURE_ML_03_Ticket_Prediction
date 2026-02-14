from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

data = pd.read_csv("resumes.csv")

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    best = None

    if request.method == "POST":
        job_desc = request.form["job"]

        documents = list(data["resume_text"]) + [job_desc]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

        data["match_score"] = scores
        results = data.sort_values(by="match_score", ascending=False)

        best = results.iloc[0]["name"]

    return render_template("index.html", tables=results, best=best)

app.run(debug=True)
