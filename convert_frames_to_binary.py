import os
import shutil

# ================= CONFIG =================
SOURCE_DIR = "frames_split"     # multi-class frames
OUTPUT_DIR = "frames_binary"    # binary frames

# ================= CREATE OUTPUT FOLDERS =================
for split in ["train", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "violence"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "non_violence"), exist_ok=True)

# ================= CONVERT TO BINARY =================
for split in ["train", "test"]:
    split_path = os.path.join(SOURCE_DIR, split)

    for label in os.listdir(split_path):
        label_path = os.path.join(split_path, label)
        if not os.path.isdir(label_path):
            continue

        # Binary rule
        if label.lower() == "normal":
            target_class = "non_violence"
        else:
            target_class = "violence"

        save_path = os.path.join(OUTPUT_DIR, split, target_class)

        for img in os.listdir(label_path):
            src = os.path.join(label_path, img)
            dst = os.path.join(save_path, f"{label}_{img}")
            shutil.move(src, dst)

        print(f"{split.upper()} | {label} → {target_class}")

print("\n✅ Binary frame conversion completed successfully!")
