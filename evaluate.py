import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIG
# =========================
IMG_SIZE = 128      # same as training
BATCH_SIZE = 32
MODEL_PATH = "brain_tumor_cnn.h5"
TEST_DIR = "data/test"

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# =========================
# LOAD TEST DATA
# =========================
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =========================
# PREDICTION
# =========================
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

class_names = list(test_data.class_indices.keys())

# =========================
# METRICS
# =========================
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n📉 Confusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))