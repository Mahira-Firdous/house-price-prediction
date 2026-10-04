"""
app.py
------
Flask REST API for house price prediction.

Endpoints
---------
GET  /health        → health check, returns {"status": "ok"}
POST /predict       → accepts JSON with 8 house features,
                      returns {"predicted_price": "$320,000"}

Start the server:
    python app.py

The server listens on http://127.0.0.1:5000 by default.
"""

from flask import Flask, jsonify, request
from src.predict import FEATURE_NAMES, predict_price

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Simple liveness check so Streamlit can verify the API is running."""
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict median house value.

    Expected JSON body (all values numeric):
    {
        "MedInc":     3.5,
        "HouseAge":   20.0,
        "AveRooms":   5.0,
        "AveBedrms":  1.0,
        "Population": 800.0,
        "AveOccup":   2.5,
        "Latitude":   34.0,
        "Longitude": -118.0
    }

    Response:
    {
        "predicted_price": "$250,000",
        "predicted_price_raw": 250000.0
    }
    """
    data = request.get_json(silent=True)

    # ── Validate request ────────────────────────────────────────────────────
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        features = {name: float(data[name]) for name in FEATURE_NAMES}
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"All feature values must be numeric. Detail: {exc}"}), 400

    # ── Predict ─────────────────────────────────────────────────────────────
    try:
        price = predict_price(features)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "predicted_price":     f"${price:,.0f}",
        "predicted_price_raw": round(price, 2),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # debug=False for a clean student experience; set to True to see tracebacks
    app.run(host="127.0.0.1", port=5000, debug=False)
