import os
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = os.path.join('Data_collection')
actions = np.array(["Hello", "Thank you", "Have a good day", "Yes", "No"])
no_sequences = 30
sequence_length = 30

# Create directories for each action
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

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(1, no_sequences + 1):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    continue
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                # Get frame dimensions
                h, w, _ = frame.shape
                if frame_num == 0:
                    # cv2.putText(image, f'Starting collection for {action} {sequence}', (120, 200),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                     # Generate text message
                    message = f'Starting collection for {action} {sequence}'
                    
                    # Calculate text size for centering
                    (text_width, text_height), baseline = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    text_x = (w - text_width) // 2
                    text_y = (h // 2) - 20

                    # Draw background rectangle for better visibility
                    cv2.rectangle(image, (text_x - 10, text_y - text_height - 10), 
                                  (text_x + text_width + 10, text_y + 10), (0, 0, 0), -1)

                    # Put centered text
                    cv2.putText(image, message, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Webcam Feed', image)
                    cv2.waitKey(1000)
                keypoints = extract_keypoints(results)
                np.save(os.path.join(DATA_PATH, action, str(sequence), str(frame_num)), keypoints)
                cv2.imshow('Webcam Feed', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()