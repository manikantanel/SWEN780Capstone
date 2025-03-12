import tensorflow as tf
from tensorflow.keras.models import load_model

print("✅ TensorFlow Version:", tf.__version__)

# Load model without compiling
try:
    model = load_model("action_1.h5", compile=False)
    print("✅ Model loaded successfully.")

    # Save a new fixed version of the model
    model.save("action_1_fixed.h5")
    print("✅ Model successfully converted and saved as 'action_1_fixed.h5'.")

except Exception as e:
    print(f"❌ Error: {e}")
