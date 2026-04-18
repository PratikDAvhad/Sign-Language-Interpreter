import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import numpy as np
import math
import time
import os

# --- CONFIGURATION ---
DATA_PATH = "Data2"
LABEL = "Z"
SAVE_PATH = os.path.join(DATA_PATH, LABEL)
IMG_SIZE = 300
OFFSET = 20

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

cap = cv2.VideoCapture(0)
# Initialize detector (Note: Some versions don't need parameters here)
detector = HandDetector(maxHands=1)

# MediaPipe utilities for drawing the skeleton
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

counter = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if not success: break

    # 1. Find the hand (REFINED: Only unpack 'hands')
    hands = detector.findHands(img, draw=False)

    if hands:
        # Depending on version, hands might be a list or a tuple.
        # This check ensures we get the first hand safely.
        hand = hands[0] if isinstance(hands, list) and len(hands) > 0 else None

        if hand:
            x, y, w, h = hand['bbox']

            # 2. Create a blank white canvas
            img_skeleton = np.ones((img.shape[0], img.shape[1], 3), np.uint8) * 255

            # 3. Draw the skeleton onto the WHITE canvas using the detector's results
            if detector.results.multi_hand_landmarks:
                for hand_lms in detector.results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        img_skeleton,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  # Red Joints
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3)  # Green Bones
                    )

            # 4. Create the final 300x300 white square
            img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

            try:
                y1, y2 = max(0, y - OFFSET), min(img.shape[0], y + h + OFFSET)
                x1, x2 = max(0, x - OFFSET), min(img.shape[1], x + w + OFFSET)
                img_crop = img_skeleton[y1:y2, x1:x2]

                # 5. Math to center the skeleton image
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

                cv2.imshow("Data Preview (The Skeleton)", img_white)
            except Exception:
                pass

    cv2.imshow("Main Webcam Feed", img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{SAVE_PATH}/Image_{time.time()}.jpg', img_white)
        print(f"Saved {counter} refined images to {SAVE_PATH}")
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()