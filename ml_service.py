"""
ml_service.py
-------------
Tiny Python Flask microservice that loads trained ML/CNN models
and serves predictions via POST /predict

Run:
    python ml_service.py
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

ACTIVITY_LABELS = {
    1:  "Standing",
    2:  "Sitting",
    3:  "Lying Down",
    4:  "Walking",
    5:  "Stairs",
    6:  "Waist Bend",
    7:  "Arm Raise",
    8:  "Crouching",
    9:  "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jump",
}

# ── Load Models ───────────────────────────────────────────
print("[ML SERVICE] Loading models...")

models = {}

# Load ML models
ml_model_files = {
    "Logistic Reg":  "Logistic_Regression.pkl",
    "KNN":           "KNN.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "Random Forest": "Random_Forest.pkl",
}

for name, filename in ml_model_files.items():
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        models[name] = joblib.load(path)
        print(f"[ML SERVICE] Loaded → {name}")
    else:
        print(f"[ML SERVICE] WARNING: {filename} not found!")

# Load scaler
scaler = None
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("[ML SERVICE] Loaded → scaler")
else:
    print("[ML SERVICE] WARNING: scaler.pkl not found!")

# Load CNN model
cnn_model = None
try:
    import tensorflow as tf
    cnn_path = os.path.join(MODELS_DIR, "CNN_1D.keras")
    if os.path.exists(cnn_path):
        cnn_model = tf.keras.models.load_model(cnn_path)
        models["1D CNN"] = "cnn"
        print("[ML SERVICE] Loaded → 1D CNN")
    else:
        print("[ML SERVICE] WARNING: CNN_1D.keras not found!")
except Exception as e:
    print(f"[ML SERVICE] CNN load failed: {e}")

print(f"[ML SERVICE] Models ready: {list(models.keys())}")

# ── Feature order ─────────────────────────────────────────
FEATURE_ORDER = ["alx", "aly", "alz", "glx", "gly", "glz",
                 "arx", "ary", "arz", "grx", "gry", "grz"]

# ── Predict endpoint ──────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data       = request.get_json()
        features   = data.get("features", {})
        model_name = data.get("model", "Random Forest")

        # Build feature array in correct order
        feature_vals = np.array(
            [features.get(f, 0.0) for f in FEATURE_ORDER]
        ).reshape(1, -1)

        # Scale features
        if scaler is not None:
            feature_vals_scaled = scaler.transform(feature_vals)
        else:
            feature_vals_scaled = feature_vals

        # ── Run prediction ────────────────────────────────
        if model_name == "1D CNN" and cnn_model is not None:
            # CNN needs shape (1, 12, 1)
            cnn_input   = feature_vals_scaled.reshape(1, 12, 1)
            pred_probs  = cnn_model.predict(cnn_input, verbose=0)
            pred_idx    = int(np.argmax(pred_probs))
            confidence  = float(np.max(pred_probs)) * 100
            # CNN classes are 0-indexed (0=activity1, 1=activity2 ...)
            activity_id = pred_idx + 1

        elif model_name in models and models[model_name] != "cnn":
            model      = models[model_name]
            pred       = model.predict(feature_vals_scaled)[0]
            activity_id = int(pred)

            # Get confidence from predict_proba if available
            if hasattr(model, "predict_proba"):
                proba      = model.predict_proba(feature_vals_scaled)[0]
                confidence = float(np.max(proba)) * 100
            else:
                confidence = 90.0

        else:
            # Fallback to Random Forest
            if "Random Forest" in models:
                model      = models["Random Forest"]
                pred       = model.predict(feature_vals_scaled)[0]
                activity_id = int(pred)
                proba      = model.predict_proba(feature_vals_scaled)[0]
                confidence = float(np.max(proba)) * 100
            else:
                return jsonify({"error": "No models available"}), 500

        activity = ACTIVITY_LABELS.get(activity_id, f"Activity {activity_id}")
        confidence = round(confidence, 2)

        print(f"[ML SERVICE] {model_name} → {activity} ({confidence}%)")

        return jsonify({
            "activity":   activity,
            "confidence": confidence,
            "model":      model_name,
            "class_id":   activity_id,
        })

    except Exception as e:
        print(f"[ML SERVICE] Error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Health check ──────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "UP",
        "models": list(models.keys()),
    })


# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("[ML SERVICE] Starting on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)