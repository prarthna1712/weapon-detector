import os
from data_preprocessing import load_data
from model import build_model

# Paths
DATA_DIR = r"dataset"   # root folder where weapons/ and nonweapons/ exist
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "valid")
TEST_DIR  = os.path.join(DATA_DIR, "test")

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = (224, 224)

# Load data
train_data, val_data = load_data(TRAIN_DIR, VAL_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

# Build model
model = build_model(input_shape=(224,224,3), num_classes=2)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",   # use sparse_categorical_crossentropy if labels are integers
    metrics=["accuracy"]
)
# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Evaluate on test set
test_ds = load_data(TEST_DIR, VAL_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)  # ⚡ small tweak here
