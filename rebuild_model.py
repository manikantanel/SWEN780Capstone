import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

print("✅ TensorFlow Version:", tf.__version__)

try:
    # Load the original model
    old_model = load_model("action_1.h5", compile=False)
    print("✅ Model loaded successfully.")

    # Define the new model (same architecture but without `time_major=False`)
    new_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(30, 258)),  # No time_major
        LSTM(128),  # No time_major
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(3, activation="softmax")
    ])

    # Copy the weights from the old model to the new one
    for layer_new, layer_old in zip(new_model.layers, old_model.layers):
        layer_new.set_weights(layer_old.get_weights())

    # Save the new fixed model
    new_model.save("action_1_rebuilt.h5")
    print("✅ Model successfully rebuilt and saved as 'action_1_rebuilt.h5'.")

except Exception as e:
    print(f"❌ Error rebuilding model: {e}")
