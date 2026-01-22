import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# ================= CONFIG =================
MODEL_PATH = "violence_binary_model.h5"
IMG_SIZE = 96

THRESHOLD = 0.6          # 🔥 tuned threshold
WINDOW_SIZE = 10         # probability smoothing
CONSEC_FRAMES = 5        # temporal consistency

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ================= VIDEO SOURCE =================
# Use 0 for webcam OR provide video path
VIDEO_SOURCE = 0  
#VIDEO_SOURCE = "test.mp4"

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("❌ Cannot open video source")
    exit()

# ================= VARIABLES =================
prob_buffer = deque(maxlen=WINDOW_SIZE)
violent_count = 0

# ================= REAL-TIME LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize & preprocess
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prob = model.predict(img, verbose=0)[0][0]
    prob_buffer.append(prob)

    # Average probability
    mean_prob = np.mean(prob_buffer)

    # Temporal logic
    if mean_prob >= THRESHOLD:
        violent_count += 1
    else:
        violent_count = 0

    # Decision
    if violent_count >= CONSEC_FRAMES:
        label = "🚨 VIOLENCE DETECTED"
        color = (0, 0, 255)
    else:
        label = "NON-VIOLENCE"
        color = (0, 255, 0)

    # Display
    cv2.putText(
        frame,
        f"{label} | Prob: {mean_prob:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Real-Time Violence Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
print("✅ Detection stopped")
