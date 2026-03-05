from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from model import build_model

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

train_dir = "data/train"
test_dir = "data/test"

# Training data generator (grayscale MRI)
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=True
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=False
)

# 🔍 Print class mapping (VERY IMPORTANT)
print("Class indices:", train_data.class_indices)

# ⚖️ Handle class imbalance
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# 🧠 Build model
model = build_model(
    num_classes=train_data.num_classes,
    input_shape=(IMG_SIZE, IMG_SIZE, 1)
)

# 🚀 Train
model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data,
    class_weight=class_weights
)

# 💾 Save model
model.save("model.h5")
print("✅ Model trained and saved as model.h5")
