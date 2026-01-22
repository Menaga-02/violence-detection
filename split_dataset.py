import os
import shutil
import random

# ================= CONFIG =================
SOURCE_DIR = "dataset"
DEST_DIR = "dataset_split"
TRAIN_RATIO = 0.8   # 80% train, 20% test

random.seed(42)

# ================= CREATE FOLDERS =================
train_dir = os.path.join(DEST_DIR, "train")
test_dir = os.path.join(DEST_DIR, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# ================= SPLIT DATA =================
for label in os.listdir(SOURCE_DIR):
    label_path = os.path.join(SOURCE_DIR, label)

    if not os.path.isdir(label_path):
        continue

    # Create label folders
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(test_dir, label), exist_ok=True)

    files = os.listdir(label_path)
    random.shuffle(files)

    split_index = int(len(files) * TRAIN_RATIO)

    train_files = files[:split_index]
    test_files = files[split_index:]

    for file in train_files:
        shutil.copy(
            os.path.join(label_path, file),
            os.path.join(train_dir, label, file)
        )

    for file in test_files:
        shutil.copy(
            os.path.join(label_path, file),
            os.path.join(test_dir, label, file)
        )

    print(f"{label}: {len(train_files)} train | {len(test_files)} test")

print("\n✅ Dataset split completed successfully!")
