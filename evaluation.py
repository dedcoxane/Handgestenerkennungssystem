import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical



DATASET_PATH = 'dataset'
MODEL_PATH = 'sign_model.h5'
CLASSES = ['Bye', 'Hello', 'No', 'Perfect', 'Thank You', 'Yes', 'A', 'B', 'C', 'D','E','F','G','H']
IMG_SIZE = 128


# Load data
def load_data():
    X, y = [], []
    for idx, label in enumerate(CLASSES):
        folder = os.path.join(DATASET_PATH, label)
        if not os.path.exists(folder):
            continue
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
                X.append(img)
                y.append(idx)
    return np.array(X), to_categorical(y)


# Load model
print("Loading data...")
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Loading model...")
model = load_model(MODEL_PATH)

# Predictions
print("\nMaking predictions...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
confidences = np.max(y_pred_probs, axis=1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print(f"\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# ===============================
# 1. CONFUSION MATRIX
# ===============================
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\n Confusion matrix saved as 'confusion_matrix.png'")

# ===============================
# 2. PREDICTED VS ACTUAL SCATTER PLOT
# ===============================
plt.figure(figsize=(10, 4))

# Scatter plot: Actual vs Predicted
plt.subplot(1, 2, 1)
for i in range(len(CLASSES)):

    indices = np.where(y_true == i)[0]
    if len(indices) > 0:

        plt.scatter([i] * len(indices), y_pred[indices],
                    alpha=0.6, label=CLASSES[i], s=50)

plt.plot([-0.5, len(CLASSES) - 0.5], [-0.5, len(CLASSES) - 0.5],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Class')
plt.ylabel('Predicted Class')
plt.title('Predicted vs Actual (Scatter)')
plt.xticks(range(len(CLASSES)), CLASSES, rotation=45)
plt.yticks(range(len(CLASSES)), CLASSES)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# ===============================
# 3. TRAINING VS VALIDATION ACCURACY GRAPH
# ===============================

try:
    # Try to load history from JSON file if it exists
    if os.path.exists('training_history.json'):
        with open('training_history.json', 'r') as f:
            history = json.load(f)
    else:

        print("\n No training history found. Using sample data for demonstration.")
        history = {
            'accuracy': [0.25, 0.45, 0.65, 0.75, 0.82, 0.87, 0.90, 0.92, 0.94, 0.95],
            'val_accuracy': [0.20, 0.40, 0.55, 0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.88]
        }

    epochs = range(1, len(history['accuracy']) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')

    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_validation_accuracy.png', dpi=100)
    print(" Training vs validation accuracy saved as 'training_validation_accuracy.png'")

except Exception as e:
    print(f" Could not create accuracy graph: {e}")

# ===============================
# 4. SAMPLE PREDICTIONS VISUALIZATION
# ===============================
plt.figure(figsize=(12, 8))
n_samples = min(12, len(X_test))
indices = np.random.choice(len(X_test), n_samples, replace=False)

for i, idx in enumerate(indices):
    plt.subplot(3, 4, i + 1)


    img = (X_test[idx] * 255).astype(np.uint8)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Get predictions
    actual_class = CLASSES[y_true[idx]]
    pred_class = CLASSES[y_pred[idx]]
    confidence = confidences[idx]


    color = 'green' if y_true[idx] == y_pred[idx] else 'red'

    plt.title(f"Actual: {actual_class}\nPred: {pred_class}\nConf: {confidence:.2f}",
              color=color, fontsize=9)
    plt.axis('off')

plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14)
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=100)
print(" Sample predictions saved as 'sample_predictions.png'")

plt.show()
print("\n Evaluation complete! Check the generated PNG files.")