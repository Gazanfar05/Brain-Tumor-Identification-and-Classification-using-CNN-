from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

train_dir = "data/train"
test_dir = "data/test"

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
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

model = build_model(num_classes=train_data.num_classes)

model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

model.save("brain_tumor_cnn.h5")
print("Model saved successfully 🧠💾")
model.save("model.h5")
print("✅ Model saved as model.h5")
