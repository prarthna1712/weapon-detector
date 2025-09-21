
from keras.models import load_model

# Load best model (already saved by ModelCheckpoint)
model = load_model("models/best_model.h5")

# Save in correct Keras format
model.save("models/best_model_saved.keras")

print("✅ Model successfully converted and saved at models/best_model_saved.keras")
