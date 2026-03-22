"""
Brain Age Prediction Web Application
=====================================
Flask server with multi-model prediction API.

Endpoints:
    GET  /                    — Serve the main page
    POST /predict             — Predict brain age (supports model selection)
    GET  /api/samples         — Get sample subjects for demo
    GET  /api/feature-info    — Get feature metadata
    GET  /api/models          — Get available models and their scores
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from model import BrainAgeModel, FEATURE_NAMES, FEATURE_DISPLAY_NAMES, FEATURE_UNITS, MODEL_INFO

from health_recommendations import generate_recommendations

app = Flask(__name__)
CORS(app)

# Initialize models on startup
print("\n" + "=" * 80)
print("  🧠  Explainable Deep Learning Framework for EEG-Based Prediction")
print("      and Visualization of Accelerated Brain Aging (DS003775)")
print("=" * 80 + "\n")
brain_model = BrainAgeModel()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict brain age from submitted features.
    
    JSON body:
    {
        "chronological_age": 55,            (optional)
        "model": "ensemble",                (optional, default: "ensemble")
        "features": { ... 25 features ... }
    }
    """
    try:
        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' in request body"}), 400

        features = data["features"]
        chronological_age = data.get("chronological_age", None)
        model_key = data.get("model", "ensemble")

        # Validate model key
        valid_models = list(MODEL_INFO.keys())
        if model_key not in valid_models:
            return jsonify({"error": f"Invalid model. Choose from: {valid_models}"}), 400

        # Validate features
        missing = [f for f in FEATURE_NAMES if f not in features]
        if missing:
            return jsonify({"error": f"Missing features: {', '.join(missing)}"}), 400

        try:
            features = {k: float(v) for k, v in features.items() if k in FEATURE_NAMES}
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid feature value: {str(e)}"}), 400

        if chronological_age is not None:
            try:
                chronological_age = float(chronological_age)
            except (ValueError, TypeError):
                chronological_age = None

        # Predict with selected model
        result = brain_model.predict_with_explanation(features, chronological_age, model_key)

        # Health recommendations
        recommendations = generate_recommendations(
            result["brain_age_gap"],
            result["feature_contributions"]
        )
        result["recommendations"] = recommendations

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/samples")
def get_samples():
    samples = brain_model.get_sample_subjects()
    return jsonify({"samples": samples})


@app.route("/api/feature-info")
def get_feature_info():
    info = []
    for fname in FEATURE_NAMES:
        info.append({
            "name": fname,
            "display_name": FEATURE_DISPLAY_NAMES[fname],
            "unit": FEATURE_UNITS[fname],
        })
    return jsonify({"features": info})


@app.route("/api/models")
def get_models():
    """Return available models and their performance metrics."""
    models = brain_model.get_available_models()
    return jsonify({"models": models})


if __name__ == "__main__":
    print("\n🌐 Starting server at http://localhost:5000")
    print("   Press Ctrl+C to stop\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
