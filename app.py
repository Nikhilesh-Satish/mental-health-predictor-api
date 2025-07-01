from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ NEW: import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # ✅ NEW: allow cross-origin requests (e.g., from React)

# Load once
model = joblib.load("random_forest_model.joblib")
column_order = joblib.load("column_order.joblib")

def preprocess_input(json_input):
    df = pd.DataFrame([json_input])
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.lower().str.replace(" ", "_", regex=False)
    df = pd.get_dummies(df)
    df = df.reindex(columns=column_order, fill_value=0)
    return df

@app.route("/")
def home():
    return {"message": "Mental Health Predictor is live!"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = preprocess_input(data)
        prediction = model.predict(df)[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
