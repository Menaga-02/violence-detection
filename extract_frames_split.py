import cv2
import os

# ================= CONFIG =================
SOURCE_DIR = "dataset_split"      # train / test videos
OUTPUT_DIR = "frames_split"

IMG_SIZE = 96
FRAME_SKIP = 5

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")

# ================= CREATE OUTPUT STRUCTURE =================
for split in ["train", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# ================= FRAME EXTRACTION =================
for split in ["train", "test"]:
    split_path = os.path.join(SOURCE_DIR, split)

    for label in os.listdir(split_path):
        label_path = os.path.join(split_path, label)
        if not os.path.isdir(label_path):
            continue

        save_label_path = os.path.join(OUTPUT_DIR, split, label)
        os.makedirs(save_label_path, exist_ok=True)

        for video in os.listdir(label_path):
            if not video.lower().endswith(VIDEO_EXT):
                continue

            video_path = os.path.join(label_path, video)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"❌ Cannot open {video}")
                continue

            frame_count = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % FRAME_SKIP == 0:
                    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                    filename = f"{video}_{saved_count}.jpg"
                    cv2.imwrite(os.path.join(save_label_path, filename), frame)
                    saved_count += 1

                frame_count += 1

            cap.release()
            print(f"{split.upper()} | {label} | {video} → {saved_count} frames")

print("\n✅ STEP 2 COMPLETED: Frames extracted (multi-class)")
