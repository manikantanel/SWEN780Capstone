from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import time

app = Flask(__name__)

# Load Model
try:
    model = load_model('action.h5')
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

actions = np.array(["Hello", "Thank you", "Have a good day"])
gesture_meanings = {
    "Hello": "A common greeting",
    "Thank you": "Expression of gratitude",
    "Have a good day": "Wishing someone well for the rest of the day"
}
sequence = []
current_word = None
webcam_active = False  # Tracks if the webcam is active
restart_requested = False  # Track if restart is needed
threshold = 0.5

# MediaPipe Setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def get_video_capture():
    """Initializes the webcam and ensures it is accessible."""
    global webcam_active
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        webcam_active = True
        return cap
    return None

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

def generate_frames():
    global sequence, current_word, webcam_active, restart_requested, predict_paused

    cap = get_video_capture()
    if not cap:
        yield b""
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        sequence.clear()
        current_word = None
        webcam_active = True
        predict_paused = False

        while webcam_active:
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            # draw_styled_landmarks(image, results)

            hands_visible = results.left_hand_landmarks or results.right_hand_landmarks

            if hands_visible and not predict_paused:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    try:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        detected_word = actions[np.argmax(res)]

                        if res[np.argmax(res)] > threshold and detected_word != current_word:
                            current_word = detected_word
                            predict_paused = True  # Pause further detection
                            print(f"✅ Detected gesture: {current_word}")
                    except Exception as pred_error:
                        print(f"❌ Prediction error: {pred_error}")

            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()
        webcam_active = False


@app.route('/')
def index():
    return render_template('index.html', current_word=current_word or "No gesture detected yet")

@app.route('/video_feed')
def video_feed():
    global restart_requested
    restart_requested = False  # Reset restart flag
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/clear_gesture')
def clear_gesture():
    global current_word, sequence, predict_paused
    current_word = None
    sequence.clear()
    predict_paused = False
    return jsonify({"message": "Gesture reset for new detection."})


@app.route('/get_current_gesture')
def get_current_gesture():
    global current_word
    meaning = gesture_meanings.get(current_word, "No gesture meaning available")
    return jsonify({
        "gesture": current_word or "No gesture detected",
        "meaning": meaning
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
