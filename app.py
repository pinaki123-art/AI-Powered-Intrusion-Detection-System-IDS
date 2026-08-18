from predict import predict_file
from flask import Flask, render_template, request
import os
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/analyze", methods=["GET", "POST"])
def analyze():

    if request.method == "POST":

        if "file" not in request.files:
            return "No file selected"

        file = request.files["file"]

        if file.filename == "":
            return "No file selected"

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        df = predict_file(filepath)

        total = len(df)

        benign = (df["Prediction"] == "BENIGN").sum()

        attack = total - benign

        return render_template(
        "result.html",
        filename=file.filename,
        total=total,
        benign=benign,
        attack=attack,
        table=df.head(20).to_html(classes="table table-striped", index=False)
       )

    return render_template("analyze.html")


if __name__ == "__main__":
    app.run(debug=True)
