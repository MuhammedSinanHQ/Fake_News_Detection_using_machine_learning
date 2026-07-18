from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    vector = vectorizer.transform([news])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        result = "✅ REAL NEWS"
    else:
        result = "❌ FAKE NEWS"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)