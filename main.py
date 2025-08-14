import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
import mediapipe as mp
from collections import deque, Counter

# -----------------------
# Set your model and encoder paths
# -----------------------
MODEL_PATH = "./best_asl_model.h5"       # <-- Set your model path here
ENCODER_PATH = "./label_encoder.pkl"     # <-- Set your label encoder path here

# Load model & label encoder
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# -----------------------
# MediaPipe setup
# -----------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="ASL Detection System", layout="wide")
st.title("🤟 ASL Sign Language Detection")
st.markdown("Real-time prediction using MediaPipe hand cropping with RGB input and smoothing.")

# Prediction smoothing queue
PREDICTION_QUEUE = deque(maxlen=7) 

def preprocess_hand(frame, landmarks):
    """Crop hand using landmarks, resize to 64x64, normalize, keep RGB"""
    h, w, _ = frame.shape
    x_coords = [lm.x * w for lm in landmarks.landmark]
    y_coords = [lm.y * h for lm in landmarks.landmark]

    # Center & box size
    x_center = int(np.mean(x_coords))
    y_center = int(np.mean(y_coords))
    box_size = int(max(max(x_coords)-min(x_coords), max(y_coords)-min(y_coords)) * 1.5)

    # Square crop
    x_min = max(x_center - box_size // 2, 0)
    x_max = min(x_center + box_size // 2, w)
    y_min = max(y_center - box_size // 2, 0)
    y_max = min(y_center + box_size // 2, h)

    hand_img = frame[y_min:y_max, x_min:x_max]

    # Resize & normalize
    hand_img = cv2.resize(hand_img, (64, 64))
    hand_img = hand_img / 255.0

    return np.expand_dims(hand_img, axis=0)  # (1,64,64,3)

# -----------------------
# Webcam
# -----------------------
run_webcam = st.checkbox("Enable Webcam", value=False)
FRAME_WINDOW = st.image([])
last_label = "No Hand"

if run_webcam:
    cap = cv2.VideoCapture(0)
    while run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not accessible")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        pred_label = last_label

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                processed_hand = preprocess_hand(frame_rgb, hand_landmarks)
                preds = model.predict(processed_hand, verbose=0)
                pred_class = np.argmax(preds)
                pred_label = label_encoder.inverse_transform([pred_class])[0]
                PREDICTION_QUEUE.append(pred_label)
                last_label = pred_label

        # Smooth prediction
        if PREDICTION_QUEUE:
            smoothed_label = Counter(PREDICTION_QUEUE).most_common(1)[0][0]
        else:
            smoothed_label = pred_label

        # Show prediction on frame
        cv2.putText(frame, f"{smoothed_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:
    st.write("Enable webcam to start prediction.")
