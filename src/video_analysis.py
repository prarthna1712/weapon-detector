import cv2
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("../models/best_model.h5")
classes = ["NonWeapon", "Weapon"]

# Updated predict_frame to return class + confidence
def predict_frame(frame):
    img = cv2.resize(frame, (224,224))
    img = np.expand_dims(img / 255.0, axis=0)
    preds = model.predict(img, verbose=0)
    class_id = np.argmax(preds)
    confidence = preds[0][class_id]
    return classes[class_id], confidence

# Start webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get prediction and confidence
    label, confidence = predict_frame(frame)

    # Choose color based on class
    color = (0,0,255) if label=="Weapon" else (0,255,0)

    # Display label + confidence
    text = f"{label}: {confidence*100:.1f}%"
    cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Optional: Draw rectangle around frame
    cv2.rectangle(frame, (5,5), (frame.shape[1]-5, frame.shape[0]-5), color, 2)

    # Show frame
    cv2.imshow("Weapon Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
