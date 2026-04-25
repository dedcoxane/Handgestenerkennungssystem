import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# SETTINGS
# ===============================
DATASET_PATH = "dataset"
CLASSES = ['Bye', 'Hello', 'No', 'Perfect', 'Thank You', 'Yes', 'A', 'B', 'C', 'D','E','F','G','H']
IMG_SIZE = 128
CHANNELS = 3

LABEL_MAP = {label: idx for idx, label in enumerate(CLASSES)}

X, y = [], []

# ===============================
# LOAD IMAGES
# ===============================
for label in CLASSES:
    folder = os.path.join(DATASET_PATH, label)

    if not os.path.exists(folder):
        print(f"Missing folder: {folder}")
        continue

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        X.append(img)
        y.append(LABEL_MAP[label])

X = np.array(X)
y = to_categorical(y)

print("Dataset shape:", X.shape)

# ===============================
# TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ===============================
# DATA AUGMENTATION
# ===============================
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# ===============================
# MODEL
# ===============================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu',
           input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(len(CLASSES), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# TRAIN
# ===============================
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(X_test, y_test)
)

# ===============================
# SAVE
# ===============================
model.save("sign_model.h5")
print(" Model saved as sign_model.h5")
