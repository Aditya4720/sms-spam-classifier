from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("vectorr.pkl", "rb") as f:
    vect = pickle.load(f)
with open("NBB.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        msg = request.form.get("message", "").strip().lower()
        if msg:
            X = vect.transform([msg])
            pred = model.predict(X)[0]
            prediction = "🚫 Spam" if pred == 1 else "✅ Not Spam"
        else:
            prediction = "⚠️ Please enter a message."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5050)
