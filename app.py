# app.py
from flask import Flask, request, render_template
import torch
import joblib
from model import FakeNewsModel
import torch.nn.functional as F  # Import for softmax

app = Flask(__name__)

# Load model
model = FakeNewsModel()
model.load_state_dict(torch.load("models/fake_news_model.pth"))
model.eval()

# Load vectorizer
vectorizer = joblib.load("models/vectorizer.pkl")

# Allowed file extensions
ALLOWED_EXTENSIONS = {"txt"}

def allowed_file(filename):
    """Check if uploaded file has a valid extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    percentage = None  # Initialize percentage

    if request.method == "POST":
        text = None

        if request.form.get("text"):
            text = request.form["text"].strip()
        elif "file" in request.files:
            file = request.files["file"]
            if file and allowed_file(file.filename):
                text = file.read().decode("utf-8").strip()
            else:
                result = "Error: Invalid file format. Only .txt files are allowed."

        if text:
            X_input = vectorizer.transform([text]).toarray()
            X_tensor = torch.tensor(X_input, dtype=torch.float32)
            # Get the raw output from the model
            output = model(X_tensor)
            # Apply softmax to get probabilities
            probabilities = F.softmax(output, dim=1)
            # Get the predicted class
            prediction = torch.argmax(probabilities, dim=1).item()
            # Get the probability of the predicted class
            percentage = probabilities[0, prediction].item() * 100
            result = "FAKE" if prediction == 0 else "REAL"

    return render_template("index.html", result=result, percentage=percentage) # Pass percentage

if __name__ == "__main__":
    app.run(debug=True)