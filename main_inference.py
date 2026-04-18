import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import numpy as np
import math
from tensorflow.keras.models import load_model
import pyttsx3

# --- CONFIGURATION ---
IMG_SIZE = 300
OFFSET = 20
MODEL_PATH = "sign_language_model.h5"
LABEL_PATH = "labels.txt"

# 1. Load the "Brain" and Labels
model = load_model(MODEL_PATH)
with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# 2. Initialize Voice Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

last_spoken = ""

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_output = img.copy()

    hands = detector.findHands(img, draw=False)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create Skeleton-on-White (Same as training data)
        img_skeleton = np.ones((img.shape[0], img.shape[1], 3), np.uint8) * 255
        if detector.results.multi_hand_landmarks:
            for hand_lms in detector.results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img_skeleton, hand_lms, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                       mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3))

        img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
        try:
            y1, y2 = max(0, y - OFFSET), min(img.shape[0], y + h + OFFSET)
            x1, x2 = max(0, x - OFFSET), min(img.shape[1], x + w + OFFSET)
            img_crop = img_skeleton[y1:y2, x1:x2]

            aspect_ratio = h / w
            if aspect_ratio > 1:
                k = IMG_SIZE / h
                w_cal = math.ceil(k * w)
                img_resize = cv2.resize(img_crop, (w_cal, IMG_SIZE))
                w_gap = math.ceil((IMG_SIZE - w_cal) / 2)
                img_white[:, w_gap:w_cal + w_gap] = img_resize
            else:
                k = IMG_SIZE / w
                h_cal = math.ceil(k * h)
                img_resize = cv2.resize(img_crop, (IMG_SIZE, h_cal))
                h_gap = math.ceil((IMG_SIZE - h_cal) / 2)
                img_white[h_gap:h_cal + h_gap, :] = img_resize

            # 3. PREDICTION
            prediction = model.predict(np.expand_dims(img_white / 255.0, axis=0), verbose=0)
            index = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > 0.8:  # Only speak if the AI is 80% sure
                current_label = labels[index]

                # Draw label on the video feed
                cv2.rectangle(img_output, (x - OFFSET, y - OFFSET - 50), (x + w + OFFSET, y - OFFSET), (0, 255, 0),
                              cv2.FILLED)
                cv2.putText(img_output, f"{current_label} ({int(confidence * 100)}%)", (x, y - 26),
                            cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)

                # 4. SPEAK
                if current_label != last_spoken:
                    engine.say(current_label)
                    engine.runAndWait()
                    last_spoken = current_label

        except Exception:
            pass

    cv2.imshow("Sign Language Interpreter", img_output)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()