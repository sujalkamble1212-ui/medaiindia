print("🔥 OPTIMIZED SKIN DISEASE TRAINING STARTED")

import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from sklearn.utils.class_weight import compute_class_weight

# ============================================
# REPRODUCIBILITY
# ============================================

np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# DATASET PATH
# ============================================

DATASET_PATH = r"D:\disease_prediction_project\SkinDisease"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# ============================================
# SETTINGS
# ============================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ============================================
# BETTER DATA AUGMENTATION
# ============================================

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    horizontal_flip=True,

    rotation_range=5,

    zoom_range=0.05
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# ============================================
# LOAD DATA
# ============================================

train_data = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ============================================
# CLASS INFO
# ============================================

num_classes = len(train_data.class_indices)

print("\n✅ NUMBER OF CLASSES:", num_classes)

print("\n✅ CLASS NAMES:")
for name, idx in train_data.class_indices.items():
    print(idx, "-->", name)

# ============================================
# CLASS WEIGHTS
# ============================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = {
    i: class_weights_array[i]
    for i in range(len(class_weights_array))
}

print("\n✅ CLASS WEIGHTS GENERATED")

# ============================================
# LOAD EFFICIENTNET
# ============================================

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# ============================================
# BUILD MODEL
# ============================================

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    BatchNormalization(),

    Dropout(0.3),

    Dense(128, activation="relu"),

    Dropout(0.2),

    Dense(num_classes, activation="softmax")
])

# ============================================
# CALLBACKS
# ============================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    ),

    ModelCheckpoint(
        "best_disease_model.keras",
        save_best_only=True,
        monitor="val_accuracy"
    )
]

# ============================================
# COMPILE MODEL
# ============================================

model.compile(
    optimizer=Adam(learning_rate=3e-5),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)

# ============================================
# PHASE 1 TRAINING
# ============================================

print("\n🚀 PHASE 1 TRAINING")

history1 = model.fit(
    train_data,

    validation_data=test_data,

    epochs=15,

    class_weight=class_weights,

    callbacks=callbacks
)

# ============================================
# FINE TUNING
# ============================================

print("\n🔥 FINE TUNING")

base_model.trainable = True

# Freeze early layers
for layer in base_model.layers[:-40]:
    layer.trainable = False

# ============================================
# RECOMPILE
# ============================================

model.compile(
    optimizer=Adam(learning_rate=1e-5),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)

# ============================================
# PHASE 2 TRAINING
# ============================================

history2 = model.fit(
    train_data,

    validation_data=test_data,

    epochs=10,

    class_weight=class_weights,

    callbacks=callbacks
)

# ============================================
# FINAL EVALUATION
# ============================================

print("\n📊 FINAL EVALUATION")

loss, accuracy = model.evaluate(test_data)

print(f"\n✅ TEST LOSS: {loss:.4f}")
print(f"✅ TEST ACCURACY: {accuracy:.4f}")

# ============================================
# SAVE FINAL MODEL
# ============================================

model.save("final_disease_model.keras")

print("\n🎉 MODEL TRAINING COMPLETE")
print("📦 SAVED AS final_disease_model.keras")