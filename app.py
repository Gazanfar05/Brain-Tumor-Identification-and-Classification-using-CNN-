from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import json
import traceback

app = Flask(__name__)

# Load model
MODEL_PATH = "model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    model = None

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
predictions_history = []

def preprocess_image(image_path):
    """Preprocess image to match model input requirements"""
    try:
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        print(f"Image shape after read: {img.shape}")
        
        # Resize to model's expected input size
        img = cv2.resize(img, (224, 224))
        print(f"Image shape after resize: {img.shape}")
        
        # Normalize to 0-1 range
        img = img.astype(np.float32) / 255.0
        
        # Add channel dimension: (224, 224) -> (224, 224, 1)
        img = np.expand_dims(img, axis=-1)
        print(f"Image shape after channel: {img.shape}")
        
        # Add batch dimension: (224, 224, 1) -> (1, 224, 224, 1)
        img = np.expand_dims(img, axis=0)
        print(f"Image shape after batch: {img.shape}")
        
        return img
    
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        raise

def calculate_metrics():
    """Calculate overall metrics from prediction history"""
    if len(predictions_history) < 2:
        return None
    
    try:
        true_labels = [p["true_label"] for p in predictions_history]
        pred_labels = [p["pred_label"] for p in predictions_history]
        
        # Get numeric indices for metrics
        true_indices = [CLASS_NAMES.index(label) for label in true_labels]
        pred_indices = [CLASS_NAMES.index(label) for label in pred_labels]
        
        accuracy = accuracy_score(true_indices, pred_indices)
        precision = precision_score(true_indices, pred_indices, average='weighted', zero_division=0)
        recall = recall_score(true_indices, pred_indices, average='weighted', zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_indices, pred_indices, labels=range(len(CLASS_NAMES)))
        
        return {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "confusion_matrix": conf_matrix.tolist(),
            "total_predictions": len(predictions_history)
        }
    except Exception as e:
        print(f"Metrics calculation error: {str(e)}")
        return None

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint"""
    temp_path = None
    
    try:
        # Check model loaded
        if model is None:
            return jsonify({"error": "Model not loaded. Check server logs."}), 500
        
        # Check file exists
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        print(f"\n{'='*50}")
        print(f"Processing file: {file.filename}")
        print(f"{'='*50}")

        # Save temporary file
        temp_path = "temp_image.jpg"
        file.save(temp_path)
        print(f"✓ File saved to {temp_path}")

        # Preprocess image
        img = preprocess_image(temp_path)
        print(f"✓ Image preprocessed")
        
        # Get predictions
        print(f"Model input shape: {model.input_shape}")
        preds = model.predict(img, verbose=0)[0]
        print(f"✓ Predictions received")
        
        print(f"\nRaw predictions: {preds}")
        print(f"Prediction sum: {np.sum(preds):.4f}")
        print(f"Max prediction: {np.max(preds):.4f}")
        print(f"Min prediction: {np.min(preds):.4f}")
        
        # Get top prediction
        idx = np.argmax(preds)
        confidence = float(preds[idx]) * 100
        label = CLASS_NAMES[idx]

        print(f"\n✓ Top prediction: {label} ({confidence:.2f}%)")

        # Build all predictions sorted by confidence
        all_predictions = []
        for i in range(len(CLASS_NAMES)):
            all_predictions.append({
                "class": CLASS_NAMES[i],
                "probability": float(preds[i]),
                "confidence": float(preds[i]) * 100
            })
        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)

        # Determine tumor status
        if label == "no_tumor":
            tumor_status = "No Tumor"
            tumor_type = None
        else:
            tumor_status = "Tumor Detected"
            tumor_type = label.replace("_", " ").title()

        response = {
            "tumor_status": tumor_status,
            "tumor_type": tumor_type,
            "confidence": round(confidence, 2),
            "all_predictions": all_predictions
        }

        print(f"\n✓ Prediction complete")
        print(f"{'='*50}\n")
        
        return jsonify(response)

    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"\n✗ ERROR: {error_msg}")
        print(traceback.format_exc())
        print(f"{'='*50}\n")
        return jsonify({"error": error_msg}), 500
    
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"✓ Temp file cleaned up")
            except:
                pass

@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Get overall model metrics"""
    try:
        metrics = calculate_metrics()
        if metrics:
            return jsonify(metrics)
        return jsonify({"error": "Insufficient data for metrics", "total": len(predictions_history)}), 200
    except Exception as e:
        print(f"Metrics error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/record-prediction", methods=["POST"])
def record_prediction():
    """Record a prediction for metrics calculation"""
    try:
        data = request.get_json()
        
        if "pred_label" not in data or "true_label" not in data:
            return jsonify({"error": "Missing prediction data"}), 400
        
        predictions_history.append({
            "pred_label": data["pred_label"],
            "true_label": data["true_label"],
            "confidence": data.get("confidence", 0)
        })
        
        metrics = calculate_metrics()
        return jsonify({
            "status": "recorded",
            "total_predictions": len(predictions_history),
            "metrics": metrics
        })
    
    except Exception as e:
        print(f"Record prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_shape": str(model.input_shape) if model else "N/A"
    })

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧠 Brain Tumor MRI Analyzer")
    print("="*50)
    print(f"Model: {MODEL_PATH}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Model loaded: {model is not None}")
    if model:
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
    print("="*50 + "\n")
    
    app.run(debug=True, host="127.0.0.1", port=5000)