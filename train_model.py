import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- CONFIGURATION ---
DATA_PATH = "Data2"
IMG_SIZE = 300  # Matches your data collection size
BATCH_SIZE = 32
# Count how many folders (letters) you have in Data/
NUM_CLASSES = len(os.listdir(DATA_PATH))

# 1. Data Preprocessing (Normalization)
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 2. Build the CNN Model Architecture
model = Sequential([
    # First Convolutional Layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    # Second Convolutional Layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Third Convolutional Layer
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Prevents overfitting
    Dense(NUM_CLASSES, activation='softmax')  # Output layer
])

# 3. Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train the Model
print("Starting Training...")
model.fit(train_generator, validation_data=val_generator, epochs=10)

# 5. Save the Model
model.save("sign_language_model.h5")
print("Model saved as sign_language_model.h5")

# Save the labels (A, B, C...) so we know which index belongs to which letter
labels = list(train_generator.class_indices.keys())
with open("labels.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")