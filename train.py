import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from pathlib import Path

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

train_dir = "data/train"
test_dir = "data/test"

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

def build_model():
    """Create CNN model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        
        # Block 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Block 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Dense layers
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def calculate_class_weights():
    """Calculate class weights to handle imbalance"""
    class_counts = {}
    total_images = 0
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))])
            class_counts[class_name] = count
            total_images += count
    
    # Calculate weights (inverse of frequency)
    class_weights = {}
    for i, class_name in enumerate(CLASS_NAMES):
        if class_counts[class_name] > 0:
            weight = total_images / (len(CLASS_NAMES) * class_counts[class_name])
            class_weights[i] = weight
    
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    print("\nClass weights (to balance training):")
    for i, weight in class_weights.items():
        print(f"  Class {i} ({CLASS_NAMES[i]}): {weight:.2f}")
    
    return class_weights

def train():
    """Train the model with class balancing"""
    print("=" * 60)
    print("Brain Tumor MRI Classification Training (BALANCED)")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists(train_dir):
        print(f"\n✗ ERROR: {train_dir} directory not found!")
        print("Please ensure you have:")
        print("  data/train/glioma/")
        print("  data/train/meningioma/")
        print("  data/train/no_tumor/")
        print("  data/train/pituitary/")
        return
    
    print(f"\n✓ Found training data directory: {train_dir}")
    
    # Calculate class weights
    class_weights = calculate_class_weights()
    
    # Create data generators with augmentation
    print("\n" + "=" * 60)
    print("Setting up data generators with augmentation...")
    print("=" * 60)
    
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Load training data
    print("\nLoading training images...")
    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training',
        seed=42
    )
    
    # Load validation data
    print("Loading validation images...")
    val_data = train_gen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation',
        seed=42
    )
    
    # Build model
    print("\n" + "=" * 60)
    print("Building model...")
    print("=" * 60)
    model = build_model()
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train with class weights
    print("\n" + "=" * 60)
    print("Starting training with class balancing...")
    print("=" * 60)
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,  # BALANCED TRAINING
        verbose=1
    )
    
    # Evaluate on test data
    print("\n" + "=" * 60)
    print("Evaluating on test data...")
    print("=" * 60)
    
    test_gen = ImageDataGenerator(rescale=1./255)
    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    test_loss, test_acc = model.evaluate(test_data, verbose=1)
    print(f"\n✓ Test Accuracy: {test_acc:.4f}")
    print(f"✓ Test Loss: {test_loss:.4f}")
    
    # Save model
    print("\n" + "=" * 60)
    print("Saving model...")
    print("=" * 60)
    model.save('model.h5')
    print("✓ Model saved to model.h5")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Epochs trained: {len(history.history['loss'])}")
    print(f"Final train accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Test predictions
    print("\n" + "=" * 60)
    print("Testing model predictions...")
    print("=" * 60)
    test_input = np.random.rand(4, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)
    predictions = model.predict(test_input, verbose=0)
    print("\nRandom image predictions:")
    for i, pred in enumerate(predictions):
        class_idx = np.argmax(pred)
        confidence = pred[class_idx] * 100
        print(f"  Image {i+1}: {CLASS_NAMES[class_idx]} ({confidence:.1f}%)")

if __name__ == '__main__':
    train()