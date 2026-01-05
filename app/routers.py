from flask import Flask, render_template, request
import numpy as np
import pickle
from keras.models import load_model
from heart_failure.utils import validate_input, log_error

app = Flask(__name__)


MODEL_PATH = "models/heart_failure_model.h5"
SCALER_PATH = "models/scaler.pkl"

model = load_model(MODEL_PATH)
scaler = pickle.load(open(SCALER_PATH, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        validate_input(data, 12)  # 12 features in our dataset
        data_np = np.array(data).reshape(1, -1)
        data_scaled = scaler.transform(data_np)
        prediction = model.predict(data_scaled)
        result = int(prediction[0][0] > 0.5)
        return render_template("index.html", prediction_text=f"Death Event Prediction: {result}")
    except Exception as e:
        log_error(e)
        return render_template("index.html", prediction_text=f"Error: {str(e)}")



