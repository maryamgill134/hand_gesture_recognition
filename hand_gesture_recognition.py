# # # hand_gesture_recognition.py

# # import os
# # import logging
# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO/WARNING
# # logging.getLogger('absl').setLevel(logging.ERROR)  # Suppress absl logs

# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # from collections import deque

# # # -----------------------------
# # # MediaPipe Initialization
# # # -----------------------------
# # mp_hands = mp.solutions.hands
# # mp_draw = mp.solutions.drawing_utils
# # hands = mp_hands.Hands(
# #     max_num_hands=2,
# #     min_detection_confidence=0.7,
# #     min_tracking_confidence=0.7
# # )

# # # -----------------------------
# # # Gesture Definitions
# # # -----------------------------
# # GESTURES = {
# #     (True, False, False, False, False): "Thumbs Up 👍",
# #     (False, True, True, False, False): "Victory ✌️",
# #     (True, True, True, True, True): "Palm Open ✋",
# #     (False, False, False, False, False): "Fist ✊",
# #     (False, True, False, False, False): "Index Up ☝️",
# #     (False, True, True, True, True): "Palm Down ✋↓",
# #     (True, True, False, False, True): "Rock 🤘"
# # }

# # # -----------------------------
# # # Per-hand gesture smoothing
# # # -----------------------------
# # gesture_histories = {}  # key=hand index, value=deque(maxlen=5)

# # # -----------------------------
# # # Function to get finger states
# # # -----------------------------
# # def get_finger_states(hand_landmarks):
# #     tips_ids = [4, 8, 12, 16, 20]
# #     states = []

# #     # Thumb (x-axis)
# #     if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0]-1].x:
# #         states.append(True)
# #     else:
# #         states.append(False)
    
# #     # Fingers (y-axis)
# #     for id in tips_ids[1:]:
# #         if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id-2].y:
# #             states.append(True)
# #         else:
# #             states.append(False)
    
# #     return states

# # # -----------------------------
# # # Check for OK gesture
# # # -----------------------------
# # def is_thumb_index_touching(hand_landmarks):
# #     thumb_tip = hand_landmarks.landmark[4]
# #     index_tip = hand_landmarks.landmark[8]
# #     distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
# #     return distance < 0.05

# # # -----------------------------
# # # Start Webcam
# # # -----------------------------
# # cap = cv2.VideoCapture(0)

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     frame = cv2.flip(frame, 1)  # Mirror
# #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #     results = hands.process(rgb_frame)

# #     if results.multi_hand_landmarks:
# #         for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
# #             # Draw landmarks
# #             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# #             # Initialize history for this hand
# #             if idx not in gesture_histories:
# #                 gesture_histories[idx] = deque(maxlen=5)

# #             # Finger states & gesture
# #             states = get_finger_states(hand_landmarks)
# #             gesture = GESTURES.get(tuple(states), "Unknown ❓")

# #             # Check for OK gesture
# #             if is_thumb_index_touching(hand_landmarks) and states[1]:
# #                 gesture = "OK 👌"

# #             # Smooth gesture per hand
# #             gesture_histories[idx].append(gesture)
# #             gesture_smoothed = max(set(gesture_histories[idx]), key=gesture_histories[idx].count)

# #             # Draw gesture label above hand
# #             h, w, _ = frame.shape
# #             x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
# #             y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h) - 20
# #             cv2.putText(frame, gesture_smoothed, (x_min, y_min),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# #             # Log gesture safely in UTF-8
# #             with open("gesture_log.txt", "a", encoding="utf-8") as f:
# #                 f.write(f"{gesture_smoothed}\n")

# #     cv2.imshow("Hand Gesture Recognition", frame)

# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()

# # hand_gesture_recognition_advanced.py

# import os
# import logging
# from datetime import datetime
# from collections import deque

# import cv2
# import mediapipe as mp
# import numpy as np

# # -----------------------------
# # Suppress TensorFlow / MediaPipe warnings
# # -----------------------------
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# logging.getLogger('absl').setLevel(logging.ERROR)

# # -----------------------------
# # MediaPipe Initialization
# # -----------------------------
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     max_num_hands=2,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )

# # -----------------------------
# # Gesture Definitions
# # -----------------------------
# GESTURES = {
#     (True, False, False, False, False): "Thumbs Up 👍",
#     (False, True, True, False, False): "Victory ✌️",
#     (True, True, True, True, True): "Palm Open ✋",
#     (False, False, False, False, False): "Fist ✊",
#     (False, True, False, False, False): "Index Up ☝️",
#     (False, True, True, True, True): "Palm Down ✋↓",
#     (True, True, False, False, True): "Rock 🤘"
# }

# # -----------------------------
# # Per-hand gesture smoothing (deque for majority voting)
# # -----------------------------
# gesture_histories = {}  # key=hand index, value=deque(maxlen=5)

# # -----------------------------
# # Finger state detection
# # -----------------------------
# def get_finger_states(hand_landmarks):
#     tips_ids = [4, 8, 12, 16, 20]
#     states = []

#     # Thumb (x-axis)
#     if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0]-1].x:
#         states.append(True)
#     else:
#         states.append(False)

#     # Fingers (y-axis)
#     for id in tips_ids[1:]:
#         if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id-2].y:
#             states.append(True)
#         else:
#             states.append(False)

#     return states

# # -----------------------------
# # OK gesture detection
# # -----------------------------
# def is_thumb_index_touching(hand_landmarks):
#     thumb_tip = hand_landmarks.landmark[4]
#     index_tip = hand_landmarks.landmark[8]
#     distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
#     return distance < 0.05

# # -----------------------------
# # Start webcam
# # -----------------------------
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(rgb_frame)

#     if results.multi_hand_landmarks:
#         for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
#             # Draw hand landmarks
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             # Initialize deque for this hand if not exists
#             if idx not in gesture_histories:
#                 gesture_histories[idx] = deque(maxlen=5)

#             # Detect finger states
#             states = get_finger_states(hand_landmarks)
#             gesture = GESTURES.get(tuple(states), "Unknown ❓")

#             # Detect OK gesture
#             if is_thumb_index_touching(hand_landmarks) and states[1]:
#                 gesture = "OK 👌"

#             # Smooth gestures (majority vote)
#             gesture_histories[idx].append(gesture)
#             gesture_smoothed = max(set(gesture_histories[idx]), key=gesture_histories[idx].count)

#             # Draw gesture label above hand
#             h, w, _ = frame.shape
#             x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
#             y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h) - 20
#             cv2.putText(frame, gesture_smoothed, (x_min, y_min),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#             # Log gesture with timestamp
#             with open("gesture_log.txt", "a", encoding="utf-8") as f:
#                 f.write(f"{datetime.now()} - {gesture_smoothed}\n")

#             # Optional: save landmarks for ML
#             # landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
#             # Save to CSV or database for training later

#     # Display webcam feed
#     cv2.imshow("Advanced Hand Gesture Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# hand_gesture_recognition_advanced.py

import os
import logging
from datetime import datetime
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# -----------------------------
# Suppress TensorFlow / MediaPipe warnings
# -----------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)

# -----------------------------
# MediaPipe Initialization
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------------
# Gesture Definitions
# -----------------------------
GESTURES = {
    (True, False, False, False, False): "Thumbs Up 👍",
    (False, True, True, False, False): "Victory ✌️",
    (True, True, True, True, True): "Palm Open ✋",
    (False, False, False, False, False): "Fist ✊",
    (False, True, False, False, False): "Index Up ☝️",
    (False, True, True, True, True): "Palm Down ✋↓",
    (True, True, False, False, True): "Rock 🤘"
}

# -----------------------------
# Per-hand gesture smoothing
# -----------------------------
gesture_histories = {}  # key=hand index, value=deque(maxlen=7)

# -----------------------------
# Finger state detection
# -----------------------------
def get_finger_states(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    states = []

    # Thumb (x-axis)
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0]-1].x:
        states.append(True)
    else:
        states.append(False)

    # Fingers (y-axis)
    for id in tips_ids[1:]:
        if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id-2].y:
            states.append(True)
        else:
            states.append(False)

    return states

# -----------------------------
# OK gesture detection
# -----------------------------
def is_thumb_index_touching(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    return distance < 0.05

# -----------------------------
# Start webcam
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Initialize deque for this hand if not exists
            if idx not in gesture_histories:
                gesture_histories[idx] = deque(maxlen=7)

            # Detect finger states
            states = get_finger_states(hand_landmarks)
            gesture = GESTURES.get(tuple(states), "Unknown ❓")

            # Detect OK gesture
            if is_thumb_index_touching(hand_landmarks) and states[1]:
                gesture = "OK 👌"

            # Smooth gestures (majority vote)
            gesture_histories[idx].append(gesture)
            gesture_smoothed = max(set(gesture_histories[idx]), key=gesture_histories[idx].count)

            # Draw gesture label above hand
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h) - 20
            cv2.putText(frame, gesture_smoothed, (x_min, y_min),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Log gesture with timestamp
            with open("gesture_log.txt", "a", encoding="utf-8") as f:
                f.write(f"{datetime.now()} - Hand {idx} - {gesture_smoothed}\n")

            # Optional: save landmarks for ML
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            filename = f"landmarks_dataset/{gesture_smoothed}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
            os.makedirs("landmarks_dataset", exist_ok=True)
            with open(filename, "w") as f:
                f.write(str(landmarks))

    # Display webcam feed
    cv2.imshow("Advanced Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
