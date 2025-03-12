from tensorflow.keras.models import load_model

# Load the existing model without compiling
model = load_model('action_1.h5', compile=False)

# Save the model again in a format compatible with TensorFlow 2.18
model.save('action_1_converted.h5', save_format='h5')

print("Model converted and saved as action_1_converted.h5")
