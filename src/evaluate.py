# src/evaluate.py
import numpy as np, os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from data_preprocessing import load_data

os.makedirs("results", exist_ok=True)

# Load datasets
train_ds, val_ds, class_names = load_data("dataset/train", "dataset/valid")

# Load model
model = tf.keras.models.load_model("models/best_model.h5")

# Load test dataset
test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(224,224),
    batch_size=32,
    label_mode="categorical",
    shuffle=False
)

# Save class names before mapping
test_class_names = test_ds_raw.class_names

# Apply preprocessing
test_ds = test_ds_raw.map(lambda x, y: (x/255.0, y))

# Evaluate
y_true = []
y_pred = []

for x, y in test_ds:
    preds = model.predict(x)
    y_true.extend(np.argmax(y.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# Classification report
print(classification_report(y_true, y_pred, target_names=test_class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.savefig("results/confusion_matrix.png")

# Save confusion matrix as CSV
pd.DataFrame(cm, index=test_class_names, columns=test_class_names).to_csv("results/confusion_matrix.csv")
