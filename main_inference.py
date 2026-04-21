import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import numpy as np
import math
from tensorflow.keras.models import load_model
import pyttsx3
import time

# --- CONFIGURATION ---
IMG_SIZE = 300
OFFSET = 20
MODEL_PATH = "sign_language_model.h5"
LABEL_PATH = "labels.txt"

IDLE_TIMEOUT = 10
SPACE_TIMEOUT = 2.0
STABILITY_THRESHOLD = 15

model = load_model(MODEL_PATH)
with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

engine = pyttsx3.init()
engine.setProperty('rate', 150)
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

current_word = ""
full_sentence = []
last_detected_char = ""
char_count = 0
last_activity_time = time.time()
last_hand_seen_time = time.time()
auto_spoken = False
space_added = True

while True:
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)
    img_output = img.copy()
    hands = detector.findHands(img, draw=False)
    current_time = time.time()

    if hands:
        last_activity_time = current_time
        last_hand_seen_time = current_time
        auto_spoken = False
        space_added = False

        hand = hands[0]
        x, y, w, h = hand['bbox']

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

            prediction = model.predict(np.expand_dims(img_white / 255.0, axis=0), verbose=0)
            index = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > 0.90:
                detected_char = labels[index]
                if detected_char == last_detected_char:
                    char_count += 1
                else:
                    char_count = 0
                    last_detected_char = detected_char

                if char_count == STABILITY_THRESHOLD:
                    current_word += detected_char
                    char_count = 0

                cv2.putText(img_output, f"Input: {detected_char}", (x, y - 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        except Exception:
            pass
    else:
        if not space_added and (current_time - last_hand_seen_time) > SPACE_TIMEOUT:
            if current_word != "":
                full_sentence.append(current_word)
                current_word = ""
            space_added = True

    idle_duration = current_time - last_activity_time
    if idle_duration > IDLE_TIMEOUT and (full_sentence or current_word) and not auto_spoken:
        final_speech = " ".join(full_sentence) + " " + current_word
        engine.say(final_speech)
        engine.runAndWait()
        full_sentence, current_word = [], ""
        auto_spoken = True

    # --- UI ---
    cv2.rectangle(img_output, (0, 0), (img.shape[1], 120), (255, 255, 255), cv2.FILLED)
    display_text = " ".join(full_sentence) + " " + current_word
    cv2.putText(img_output, display_text, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    if (full_sentence or current_word) and not hands:
        if not space_added:
            s_timer = max(0, int(SPACE_TIMEOUT - (current_time - last_hand_seen_time)))
            cv2.putText(img_output, f"Space in: {s_timer}s", (20, 85), 1, 1.5, (255, 165, 0), 2)
        sp_timer = max(0, int(IDLE_TIMEOUT - idle_duration))
        cv2.putText(img_output, f"Auto-speak: {sp_timer}s", (250, 85), 1, 1.5, (0, 0, 255), 2)

    cv2.imshow("Sign Language System", img_output)

    key = cv2.waitKey(1)
    if key != -1:
        last_activity_time = time.time()
        auto_spoken = False

    if key == ord(" "):  # Manual Space
        if current_word != "":
            full_sentence.append(current_word)
            current_word = ""
            space_added = True

    if key == ord("\r"):  # Enter
        final_speech = " ".join(full_sentence) + " " + current_word
        engine.say(final_speech)
        engine.runAndWait()
        full_sentence, current_word = [], ""

    if key == ord("\b"):  # Smart Backspace
        if current_word != "":
            current_word = current_word[:-1]
        elif full_sentence:
            current_word = full_sentence.pop()
        space_added = False

    if key == ord("q"): break

cap.release()
cv2.destroyAllWindows()