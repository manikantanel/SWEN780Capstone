import tensorflow as tf
from tensorflow.keras.models import load_model

print("✅ TensorFlow Version:", tf.__version__)

try:
    model = load_model("action_1.h5", compile=False)
    model.summary()  # Print model architecture
except Exception as e:
    print(f"❌ Error loading model: {e}")
