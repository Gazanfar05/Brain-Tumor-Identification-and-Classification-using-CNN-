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
    # Read image in grayscale because model expects 1 channel
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Could not read the image file")

    # Resize to model input size
    img = cv2.resize(img, (224, 224))

    # Normalize
    img = img / 255.0

    # Add channel dimension: (224, 224, 1)
    img = np.expand_dims(img, axis=-1)

    # Add batch dimension: (1, 224, 224, 1)
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

    try:
        img = preprocess_image(temp_path)
        preds = model.predict(img)[0]
        print("Preds:", preds)   
    except Exception as e:
        os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

    os.remove(temp_path)

    idx = np.argmax(preds)
    label = CLASS_NAMES[idx]
    confidence = round(float(preds[idx]) * 100, 2)

    if label == "no_tumor":
        tumor_status = "No Tumor"
        tumor_type = None
    else:
        tumor_status = "Tumor Detected"
        tumor_type = label.capitalize()

    return jsonify({
        "tumor_status": tumor_status,
        "tumor_type": tumor_type,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)
