from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models & precomputed results
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
cluster_profile = joblib.load("cluster_profile.pkl")
cluster_insights = joblib.load("cluster_insights.pkl")
elasticity_summary = joblib.load("elasticity_summary.pkl")
price_simulations = joblib.load("price_simulations.pkl")

# ------------------- Endpoints -------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "LG Segmentation & Pricing API is running ✅",
        "endpoints": {
            "POST /predict": "Predict customer cluster",
            "GET /clusters/profile": "Cluster profiles",
            "GET /clusters/insights": "Cluster insights",
            "GET /pricing/elasticity": "Elasticity summary",
            "GET /pricing/simulations": "Price simulations"
        }
    })

@app.route("/predict", methods=["POST"])
def predict_cluster():
    data = request.json
    features = np.array([[
        data["Age"],
        data["Income"],
        data["LoyaltyScore"],
        data["OnlineEngagement"],
        data["DaysSinceLastPurchase"],
        data["QuantityPurchased"],
        data["PreferenceScore"],
        data["WillingnessToPay"]
    ]])
    scaled = scaler.transform(features)
    cluster = int(kmeans.predict(scaled)[0])
    return jsonify({"cluster": cluster})

@app.route("/clusters/profile", methods=["GET"])
def get_profile():
    return jsonify(cluster_profile.to_dict(orient="index"))

@app.route("/clusters/insights", methods=["GET"])
def get_insights():
    return jsonify(cluster_insights.to_dict(orient="index"))

@app.route("/pricing/elasticity", methods=["GET"])
def get_elasticity():
    return jsonify(elasticity_summary.to_dict(orient="records"))

@app.route("/pricing/simulations", methods=["GET"])
def get_simulations():
    return jsonify(price_simulations.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
