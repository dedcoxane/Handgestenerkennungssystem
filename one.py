import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator








import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ===============================
# SETTINGS
# ===============================
MODEL_PATH = "sign_model.h5"
IMAGE_PATH = "dataset/Bye/Image_1667239052.882727.jpg"

CLASSES = ['Bye', 'Hello', 'No', 'Perfect', 'Thank You', 'Yes', 'A', 'B', 'C', 'D']
IMG_SIZE = 128

# ===============================
# LOAD MODEL
# ===============================
model = load_model(MODEL_PATH)

# ===============================
# LOAD IMAGE
# ===============================
img = cv2.imread(IMAGE_PATH)

if img is None:
    print("❌ Image not found or cannot be read")
    exit()

# Resize and normalize (SAME as training)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0

# Add batch dimension
img = np.expand_dims(img, axis=0)

# ===============================
# PREDICTION
# ===============================
prediction = model.predict(img, verbose=0)

class_id = np.argmax(prediction)
confidence = np.max(prediction)

print("Raw prediction vector:")
print(prediction)

print("\nPredicted class:", CLASSES[class_id])
print("Confidence:", confidence)
