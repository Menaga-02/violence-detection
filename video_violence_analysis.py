import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# ================= CONFIG =================
MODEL_PATH = "violence_binary_model.h5"
VIDEO_PATH = "test.mp4"   # 👈 change this

IMG_SIZE = 96

THRESHOLD = 0.6                  # tuned threshold
FRAME_SKIP = 5                   # same as training
WINDOW_SIZE = 10                 # smoothing window
CONSEC_FRAMES = 10               # consecutive violent frames

MIN_VIOLENCE_SECONDS = 2         # 🔥 key fix for crowd videos

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# ================= LOAD VIDEO =================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

FPS = int(cap.get(cv2.CAP_PROP_FPS))
MIN_VIOLENCE_FRAMES = FPS * MIN_VIOLENCE_SECONDS

print(f"🎥 FPS: {FPS}")
print(f"⏱ Violence required: {MIN_VIOLENCE_SECONDS} seconds")

# ================= VARIABLES =================
prob_buffer = deque(maxlen=WINDOW_SIZE)
violent_streak = 0
total_violent_frames = 0
frame_count = 0

# ================= PROCESS VIDEO =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames (speed + consistency)
    if frame_count % FRAME_SKIP != 0:
        continue

    # Preprocess frame
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prob = model.predict(img, verbose=0)[0][0]
    prob_buffer.append(prob)

    # Smoothed probability
    mean_prob = np.mean(prob_buffer)

    # Temporal logic
    if mean_prob >= THRESHOLD:
        violent_streak += 1
        total_violent_frames += 1
    else:
        violent_streak = 0

    # Live status
    if violent_streak >= CONSEC_FRAMES:
        label = "VIOLENCE (ongoing)"
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

    cv2.imshow("Video Violence Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ================= FINAL DECISION =================
print("\n================ FINAL RESULT ================")
print(f"Total violent frames: {total_violent_frames}")
print(f"Required frames     : {MIN_VIOLENCE_FRAMES}")

if total_violent_frames >= MIN_VIOLENCE_FRAMES:
    print("🚨 FINAL OUTPUT: VIOLENCE VIDEO")
else:
    print("✅ FINAL OUTPUT: NON-VIOLENCE VIDEO")
