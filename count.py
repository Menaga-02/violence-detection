# import os

# # ================= CONFIG =================
# TEST_DIR = "frames_binary/test"
# # VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")
# VIDEO_EXT = (".jpg")

# # ================= COUNT =================
# print("\n📊 VIDEO COUNT IN TEST DATASET")

# total_test = 0

# for cls in ["violence", "non_violence"]:
#     class_path = os.path.join(TEST_DIR, cls)
    
#     videos = [
#         f for f in os.listdir(class_path)
#         if f.lower().endswith(VIDEO_EXT)
#     ]
    
#     count = len(videos)
#     total_test += count

#     print(f"{cls.capitalize():14}: {count}")

# print(f"\nTotal Test Videos  : {total_test}")
# print("\n✅ Counting completed")

import os

# ================= CONFIG =================
TEST_DIR = "frames_binary/train"
# VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")
VIDEO_EXT = (".jpg")

# ================= COUNT =================
print("\n📊 VIDEO COUNT IN TRAIN DATASET")

total_test = 0

for cls in ["violence", "non_violence"]:
    class_path = os.path.join(TEST_DIR, cls)
    
    videos = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(VIDEO_EXT)
    ]
    
    count = len(videos)
    total_test += count

    print(f"{cls.capitalize():14}: {count}")

print(f"\nTotal Test Videos  : {total_test}")
print("\n✅ Counting completed")
