import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# ================= CONFIG =================
IMG_SIZE = 96
BATCH_SIZE = 32
THRESHOLD = 0.55   # 🔥 start here

TEST_DIR = "frames_binary/test"
MODEL_PATH = "violence_binary_model.h5"

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# ================= LOAD TEST DATA =================
datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ================= PREDICT =================
probs = model.predict(test_data).ravel()
y_pred = (probs >= THRESHOLD).astype(int)
y_true = test_data.classes

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# ================= METRICS CALCULATION =================
accuracy = (tp + tn) / (tp + tn + fp + fn)

report = classification_report(
    y_true,
    y_pred,
    output_dict=True
)

precision = report["weighted avg"]["precision"]
recall = report["weighted avg"]["recall"]
f1_score = report["weighted avg"]["f1-score"]


# ================= FINAL OUTPUT =================
print("\n📊 EXPERIMENTAL RESULTS (PERCENTAGE)")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1_score * 100:.2f}%")

print("\n📊 CONFUSION MATRIX")
print(cm)

