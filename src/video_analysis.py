
import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("../models/best_model.h5")
classes = ["NonWeapon", "Weapon"]

def predict_frame(frame):
    img = cv2.resize(frame, (224,224))
    img = np.expand_dims(img / 255.0, axis=0)
    preds = model.predict(img)
    return classes[np.argmax(preds)]

cap = cv2.VideoCapture(0)  # webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    label = predict_frame(frame)
    cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Weapon Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
