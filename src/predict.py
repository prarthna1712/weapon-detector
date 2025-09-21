# src/predict.py
import os, argparse, cv2, numpy as np, csv, time
from keras.models import load_model

# Reduce TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("--video", default=None, help="Path to video file, default webcam")
parser.add_argument("--out", default="results/out.mp4", help="Path to save output video")
parser.add_argument("--skip", type=int, default=3, help="Process every Nth frame")
args = parser.parse_args()

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Load model
model = load_model("models/best_model.h5")
class_names = ["no_weapon","weapon"]

# Open video source
cap = cv2.VideoCapture(0 if args.video is None else args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_vid = cv2.VideoWriter(args.out, fourcc, fps, (w,h))

# Log file
with open("results/detections.csv","w",newline="") as log:
    writer = csv.writer(log)
    writer.writerow(["frame","time","pred","conf"])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.skip == 0:
            img = cv2.resize(frame, (224,224)).astype("float32")/255.0
            pred = model.predict(np.expand_dims(img,0), verbose=0)[0]
            idx = int(pred.argmax())
            conf = float(pred[idx])
            label = f"{class_names[idx]} {conf:.2f}"
            color = (0,255,0) if idx==0 else (0,0,255)
            cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if idx == 1:  # weapon detected
                cv2.rectangle(frame,(5,5),(w-5,h-5), (0,0,255), 6)

            # log
            writer.writerow([frame_idx, time.time(), class_names[idx], conf])

        out_vid.write(frame)
        cv2.imshow("Detect", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1

cap.release()
out_vid.release()
cv2.destroyAllWindows()
