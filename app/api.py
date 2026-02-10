import os
import sys
from flask import Flask, request, jsonify

# Ensure 'src' is importable when running "python app/api.py"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import predict

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' (non-empty string) is required"}), 400

    result = predict(text)
    return jsonify(result), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "distilbert-base-uncased"}), 200


if __name__ == "__main__":
    # For local run: python app/api.py
    import os
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
