from flask import Flask, render_template, Response, request, redirect, url_for, send_file
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
import pytz
import csv
import os
import io
from collections import deque

# ================= APP SETUP =================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads/videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = 96
tz = pytz.timezone("Asia/Kolkata")

# ================= LOAD MODEL =================
binary_model = tf.keras.models.load_model("violence_binary_model.h5")

# ================= CONFIG (LOCK THESE) =================
FRAME_SKIP = 5
THRESHOLD = 0.6
WINDOW_SIZE = 10
CONSEC_FRAMES = 10
MIN_VIOLENCE_SECONDS = 2
VIOLENCE_RATIO_THRESHOLD = 0.15
CROWD_MOTION_THRESHOLD = 2.0

# ================= STORAGE =================
realtime_events = []
video_events = []

realtime_status = "Idle"
video_status = "Idle"
stop_realtime = False

# ================= REAL-TIME CAMERA =================
def generate_camera():
    global stop_realtime, realtime_status

    cap = cv2.VideoCapture(0)
    prob_buffer = deque(maxlen=WINDOW_SIZE)
    violent_count = 0
    violence_active = False
    start_time = None
    confidence = 0.0

    while not stop_realtime:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        prob = float(binary_model.predict(img, verbose=0)[0][0])
        prob_buffer.append(prob)
        mean_prob = np.mean(prob_buffer)

        if mean_prob >= THRESHOLD:
            violent_count += 1
        else:
            violent_count = 0

        if violent_count >= CONSEC_FRAMES:
            realtime_status = "Violence"
            color = (0, 0, 255)
            text = "VIOLENCE"

            if not violence_active:
                violence_active = True
                start_time = datetime.now(tz)
                confidence = mean_prob
        else:
            realtime_status = "Non-Violence"
            color = (0, 255, 0)
            text = "Non-Violence"

            if violence_active:
                end_time = datetime.now(tz)
                realtime_events.append([
                    start_time.strftime("%Y-%m-%d"),
                    start_time.strftime("%H:%M:%S"),
                    end_time.strftime("%H:%M:%S"),
                    "Violence",
                    round(confidence, 3)
                ])
                violence_active = False

        cv2.putText(frame, f"{text} | {mean_prob:.2f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")

    cap.release()
    stop_realtime = False

# ================= VIDEO ANALYSIS =================
def analyze_video(path, video_name):
    global video_status, video_events

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    effective_fps = fps / FRAME_SKIP
    min_violence_frames = int(effective_fps * MIN_VIOLENCE_SECONDS)

    prob_buffer = deque(maxlen=WINDOW_SIZE)
    total_violent_frames = 0
    frame_count = 0

    prev_gray = None
    motion_sum = 0
    motion_frames = 0

    now = datetime.now(tz)
    date_ist = now.strftime("%Y-%m-%d")
    sys_time = now.strftime("%H:%M:%S")

    video_status = "Analyzing"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_sum += np.mean(mag)
            motion_frames += 1

        prev_gray = gray

        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        prob = float(binary_model.predict(img, verbose=0)[0][0])
        prob_buffer.append(prob)
        mean_prob = np.mean(prob_buffer)

        if mean_prob >= THRESHOLD:
            total_violent_frames += 1

    cap.release()

    total_processed_frames = max(1, frame_count // FRAME_SKIP)
    violence_ratio = total_violent_frames / total_processed_frames
    avg_motion = motion_sum / max(1, motion_frames)

    video_time = str(timedelta(seconds=int(frame_count / fps)))

    # ================= FINAL DECISION =================
    if (
        total_violent_frames >= min_violence_frames
        and violence_ratio >= VIOLENCE_RATIO_THRESHOLD
        and avg_motion < CROWD_MOTION_THRESHOLD
    ):
        video_events.append([
            video_name,
            date_ist,
            sys_time,
            video_time,
            "00:00:00",
            video_time,
            "Violence",
            round(mean_prob, 3)
        ])
        video_status = "Violence"
    else:
        video_events.append([
            video_name,
            date_ist,
            sys_time,
            video_time,
            "00:00:00",
            "00:00:00",
            "Non-Violence",
            1.0
        ])
        video_status = "Non-Violence"

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_realtime")
def start_realtime():
    global stop_realtime
    stop_realtime = False
    return Response(generate_camera(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stop_realtime")
def stop_realtime_route():
    global stop_realtime
    stop_realtime = True
    return redirect(url_for("index"))

@app.route("/upload_video", methods=["POST"])
def upload_video():
    file = request.files["video"]
    if file:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        analyze_video(path, file.filename)
    return redirect(url_for("index"))

@app.route("/download_realtime")
def download_realtime():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Start Time", "End Time", "Status", "Confidence"])
    writer.writerows(realtime_events)

    mem = io.BytesIO(output.getvalue().encode())
    mem.seek(0)
    return send_file(mem, as_attachment=True,
                     download_name="realtime_report.csv")

@app.route("/download_video")
def download_video():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Video Name",
        "Date (IST)",
        "System Time (IST)",
        "Video Timestamp",
        "Video Start",
        "Video End",
        "Status",
        "Confidence"
    ])
    writer.writerows(video_events)

    mem = io.BytesIO(output.getvalue().encode())
    mem.seek(0)
    return send_file(mem, as_attachment=True,
                     download_name="video_report.csv")

@app.route("/realtime_status")
def realtime_status_api():
    return {"status": realtime_status}

@app.route("/video_status")
def video_status_api():
    return {"status": video_status}

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
