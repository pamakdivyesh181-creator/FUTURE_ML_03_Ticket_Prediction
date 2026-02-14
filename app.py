from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
data = pd.read_csv("dataset.csv")

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    best_candidate = None

    if request.method == "POST":
        job_desc = request.form["job"]

        # Combine resumes + job description
        documents = list(data["resume_text"]) + [job_desc]

        # TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Similarity
        scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

        data["match_score"] = scores

        # Sort
        results = data.sort_values(by="match_score", ascending=False)

        # Best candidate
        best_candidate = results.iloc[0]["name"]

    return render_template("index.html", tables=results, best=best_candidate)

if __name__ == "__main__":
    app.run(debug=True)
