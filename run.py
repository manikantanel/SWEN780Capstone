import numpy as np
import cv2
from tensorflow.keras.models import load_model
import mediapipe as mp
# from app import app

# if __name__ == "__main__":
#     app.run(debug=True)


# Load trained model
model = load_model('action_3.h5')

# Define gestures/actions
actions = np.array(["Hello", "Thank you", "Have a good day"])

# Initialize variables
sequence = []
current_word = None
threshold = 0.5  # Confidence threshold
last_detected_word = None  # Store last displayed word

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """Processes an image using Mediapipe holistic model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def draw_styled_landmarks(image, results):
    """Draws landmarks for pose and hands."""
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    """Extracts pose and hand keypoints from Mediapipe results."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

# Open webcam
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        # **Check if hands are present before prediction**
        hands_visible = results.left_hand_landmarks or results.right_hand_landmarks

        # Extract keypoints if hands are visible
        if hands_visible:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Keep only the last 30 frames

            # Make prediction when enough frames are collected
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]

                # Identify the most confident gesture
                max_index = np.argmax(res)
                max_confidence = res[max_index]

                # Ensure **only one word is displayed at a time** and avoid flickering
                if max_confidence != threshold:
                    detected_word = actions[max_index]
                    
                    # Only update if the detected word is different from the last detected word
                    if detected_word != last_detected_word:
                        current_word = detected_word
                        last_detected_word = detected_word  # Store last detected word
                else:
                    current_word = None  # No confident gesture, display blank
                    last_detected_word = None  # Reset last detected word
        else:
            # **Force blank screen if no hands are detected**
            current_word = None
            last_detected_word = None  # Reset last detected word

        # Display detected word or blank screen
        display_text = current_word if current_word else ''
        cv2.putText(image, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the webcam feed
        cv2.imshow('Real-Time Sign Detection', image)

        # Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
