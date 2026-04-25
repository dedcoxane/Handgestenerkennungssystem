

# 🤟 Sign Language Recognition System

A real-time sign language detection system built for the **deaf and hard-of-hearing community**. It uses a trained CNN model and MediaPipe to detect hand gestures via webcam and display them as readable text on screen.

---

## 📁 Project Structure

```
main/
├── dataset/
│   ├── Bye/
│   ├── Hello/
│   ├── No/
│   ├── Perfect/
│   ├── Thank You/
│   ├── Yes/
│   ├── A/
│   ├── B/
│   ├── C/
│   ├── D/
│   ├── E/
│   ├── F/
│   ├── G/
│   └── H/
├── train.py          # Train the CNN model
├── predict.py        # Real-time webcam prediction
├── evaluation.py     # Evaluate model performance & generate plots
└── sign_model.h5     # Saved model (generated after training)
```

---

## 🧠 Supported Classes

| Gestures | Letters |
|----------|---------|
| Bye, Hello, No, Perfect, Thank You, Yes | A, B, C, D, E, F, G, H |

---

## ⚙️ Requirements

### Install dependencies

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy
opencv-python
scikit-learn
tensorflow
matplotlib
seaborn
mediapipe
```

---

## 🚀 How to Run

### Step 1 — Prepare Dataset

Place your images inside `dataset/` with one subfolder per class:

```
dataset/
  Hello/
    img1.jpg
    img2.jpg
    ...
  Bye/
    ...
```

---

### Step 2 — Train the Model

Run `train.py` to train the CNN on your dataset. This will:

- Load and preprocess all images (resized to 128×128)
- Apply data augmentation (rotation, zoom, flipping)
- Train a CNN for 20 epochs
- Save the trained model as `sign_model.h5`

```bash
python train.py
```

---

### Step 3 — Run Real-Time Prediction

Run `predict.py` to launch the webcam-based sign language detector. It will:

- Detect your hand using MediaPipe
- Crop and feed the hand region into the model
- Display the predicted sign and confidence score on screen

```bash
python predict.py
```

> Press **`Q`** to quit the webcam window.

---

### (Optional) Step 4 — Evaluate the Model

Run `evaluation.py` to generate performance metrics and visualizations:

```bash
python evaluation.py
```

This generates:

| File | Description |
|------|-------------|
| `confusion_matrix.png` | Heatmap of predicted vs actual classes |
| `training_validation_accuracy.png` | Accuracy curves over epochs |
| `sample_predictions.png` | Sample test images with predictions |

---

## 🏗️ Model Architecture

```
Input (128×128×3)
  → Conv2D(32) + MaxPooling
  → Conv2D(64) + MaxPooling
  → Conv2D(128) + MaxPooling
  → Flatten
  → Dense(256) + Dropout(0.5)
  → Softmax Output (14 classes)
```

Trained with the **Adam optimizer** and **categorical crossentropy** loss.

---

## 📊 Notes

- Training history is saved as `training_history.json` if you modify `train.py` to export it — otherwise `evaluation.py` uses sample data for the accuracy graph.
- The model input size is fixed at **128×128 pixels**.
- MediaPipe detects up to **1 hand** per frame during prediction.

---

## 🤝 Contributing

This project was built to assist the deaf and hard-of-hearing community. Contributions, new gesture classes, and dataset improvements are welcome!


Author: Mohammad Ahmad Khan 
Institution: Frankfurt University of Applied Sciences 
Semester: Winter 2025/2026
