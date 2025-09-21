import os, matplotlib.pyplot as plt
from model import build_model
from data_preprocessing import load_data
import tensorflow as tf

# Paths
TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/valid"
IMG_SIZE  = (224,224)
BATCH = 32
EPOCHS = 10

# Load data
train_ds, val_ds, class_names = load_data(TRAIN_DIR, VAL_DIR, img_size=IMG_SIZE, batch_size=BATCH)

# Build model
model = build_model(input_shape=(*IMG_SIZE,3), num_classes=len(class_names))

# ✅ Use sparse_categorical_crossentropy since labels are int
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Checkpoint & callbacks
os.makedirs("models", exist_ok=True)
ckpt = tf.keras.callbacks.ModelCheckpoint("models/best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")
es   = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)
rlr  = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)

# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[ckpt, es, rlr])

# Save final SavedModel
# model.save("models/best_model_saved.keras")


# Save training curves
os.makedirs("results", exist_ok=True)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend(); plt.savefig("results/train_accuracy.png"); plt.clf()

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend(); plt.savefig("results/train_loss.png")
