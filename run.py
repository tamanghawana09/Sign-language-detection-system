import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
import mediapipe as mp

# -----------------------
# Load model & encoder
# -----------------------
MODEL_PATH = "./best_asl_model.h5"
ENCODER_PATH = "./label_encoder.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# -----------------------
# MediaPipe setup
# -----------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="ASL Sign Detection", layout="wide")
st.title("🤟 ASL Sign Language Detection")
st.markdown("Real-time prediction with MediaPipe hand cropping.")

def preprocess_hand(frame, landmarks):
    h, w, _ = frame.shape
    x_coords = [lm.x * w for lm in landmarks.landmark]
    y_coords = [lm.y * h for lm in landmarks.landmark]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    # Padding
    pad = 20
    x_min = max(x_min - pad, 0)
    y_min = max(y_min - pad, 0)
    x_max = min(x_max + pad, w)
    y_max = min(y_max + pad, h)

    hand_img = frame[y_min:y_max, x_min:x_max]
    hand_img = cv2.resize(hand_img, (64, 64))
    hand_img = hand_img / 255.0
    return np.expand_dims(hand_img, axis=0)

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
                preds = model.predict(processed_hand)
                pred_class = np.argmax(preds)
                pred_label = label_encoder.inverse_transform([pred_class])[0]
                last_label = pred_label

        cv2.putText(frame, f"{pred_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:
    st.write("Enable webcam to start prediction.")
