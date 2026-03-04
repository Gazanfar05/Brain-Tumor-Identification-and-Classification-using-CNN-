from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load model (keep model.h5 locally, NOT on GitHub)
MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

def preprocess_image(image_path):
    # Get expected input size from the model itself
    _, height, width, channels = model.input_shape

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Could not read the image file")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    img = img / 255.0

    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    temp_path = "temp.jpg"
    file.save(temp_path)

    img = preprocess_image(temp_path)
    preds = model.predict(img)[0]

    os.remove(temp_path)

    idx = np.argmax(preds)
    confidence = float(preds[idx]) * 100

    return jsonify({
        "prediction": CLASS_NAMES[idx],
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)