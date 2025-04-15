import os
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = os.path.join('Data_collection')
actions = np.array(["Hello", "Thank you", "Have a good day"])
no_sequences = 30
sequence_length = 30

# Create folders
for action in actions:
    for sequence in range(1, no_sequences + 1):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_landmarks(image, results):
    h, w, _ = image.shape

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        x1 = int(min(lm[mp_holistic.PoseLandmark.LEFT_SHOULDER].x, lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x) * w)
        y1 = int(min(lm[mp_holistic.PoseLandmark.LEFT_SHOULDER].y, lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y) * h)
        x2 = int(max(lm[mp_holistic.PoseLandmark.LEFT_HIP].x, lm[mp_holistic.PoseLandmark.RIGHT_HIP].x) * w)
        y2 = int(max(lm[mp_holistic.PoseLandmark.LEFT_HIP].y, lm[mp_holistic.PoseLandmark.RIGHT_HIP].y) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(image, 'Chest', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    if results.left_hand_landmarks:
        coords = [(int(lm.x * w), int(lm.y * h)) for lm in results.left_hand_landmarks.landmark]
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        cv2.putText(image, 'Left Hand', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    if results.right_hand_landmarks:
        coords = [(int(lm.x * w), int(lm.y * h)) for lm in results.right_hand_landmarks.landmark]
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 165, 255), 2)
        cv2.putText(image, 'Right Hand', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

def save_prediction_to_file(prediction_text, file_path='gesture_predictions.txt'):
    with open(file_path, 'a') as file:
        file.write(f"{prediction_text}\n")

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 4))
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose.flatten(), lh.flatten(), rh.flatten()])

cap = cv2.VideoCapture(0)
paused = False

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(1, no_sequences + 1):
            frame_num = 0
            while frame_num < sequence_length:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    draw_landmarks(image, results)

                    h, w, _ = frame.shape
                    if frame_num == 0:
                        msg = f"Starting collection for {action} {sequence}"
                        save_prediction_to_file(msg)
                        (text_w, text_h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        tx = (w - text_w) // 2
                        ty = h // 2
                        cv2.rectangle(image, (tx - 10, ty - text_h - 10),
                                      (tx + text_w + 10, ty + 10), (0, 0, 0), -1)
                        cv2.putText(image, msg, (tx, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Webcam Feed', image)
                        cv2.waitKey(1000)

                    # Save keypoints
                    keypoints = extract_keypoints(results)
                    np.save(os.path.join(DATA_PATH, action, str(sequence), str(frame_num)), keypoints)
                    frame_num += 1

                    cv2.imshow('Webcam Feed', image)

                # Keyboard controls
                key = cv2.waitKey(10) & 0xFF
                if key == ord('p'):
                    paused = not paused
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

cap.release()
cv2.destroyAllWindows()
