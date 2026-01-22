import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ================= CONFIG =================
IMG_SIZE = 96
BATCH_SIZE = 32

TEST_DIR = "frames_binary/test"
MODEL_PATH = "violence_binary_model.h5"

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# ================= DATA =================
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ================= EVALUATE =================
loss, acc = model.evaluate(test_data)

print("\n📊 TEST RESULTS")
print(f"Testing Accuracy : {acc*100:.2f}%")
print(f"Testing Loss     : {loss:.4f}")
