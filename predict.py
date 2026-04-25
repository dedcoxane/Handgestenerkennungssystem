import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ===============================
# SETTINGS
# ===============================
MODEL_PATH = "sign_model.h5"
CLASSES = ['Bye', 'Hello', 'No', 'Perfect', 'Thank You', 'Yes', 'A', 'B', 'C', 'D','E','F','G','H']
IMG_SIZE = 128

# ===============================
# LOAD MODEL
# ===============================
model = load_model(MODEL_PATH)

# ===============================
# MEDIAPIPE
# ===============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===============================
# WEBCAM
# ===============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        x_list, y_list = [], []
        for lm in hand_landmarks.landmark:
            x_list.append(int(lm.x * w))
            y_list.append(int(lm.y * h))

        x_min = max(min(x_list) - 20, 0)
        x_max = min(max(x_list) + 20, w)
        y_min = max(min(y_list) - 20, 0)
        y_max = min(max(y_list) + 20, h)

        hand_img = frame[y_min:y_max, x_min:x_max]

        if hand_img.size > 0:
            hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            prediction = model.predict(hand_img, verbose=0)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)

            label = f"{CLASSES[class_id]} ({confidence:.2f})"

            cv2.putText(
                frame,
                label,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
