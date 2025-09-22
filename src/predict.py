# src/predict.py
import os, argparse, cv2, numpy as np, csv, time
from keras.models import load_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("--video", default=None, help="Path to video file, default webcam")
parser.add_argument("--out", default="results/out.mp4", help="Path to save output video")
parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame")
args = parser.parse_args()

os.makedirs("results", exist_ok=True)

# Load model
model = load_model("models/best_model.h5")

# ⚠️ Make sure this matches your training dataset order
class_names = ["no_weapon", "weapon"]

# Adjust this to your training image size (224,224) or whatever you used
IMG_SIZE = (224,224)

# Open video
cap = cv2.VideoCapture(0 if args.video is None else args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_vid = cv2.VideoWriter(args.out, fourcc, fps, (w,h))

# CSV log
log = open("results/detections.csv","w",newline="")
writer = csv.writer(log)
writer.writerow(["frame","time","pred","conf","no_weapon_prob","weapon_prob"])

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % args.skip == 0:
        # Preprocess
        img = cv2.resize(frame, IMG_SIZE).astype("float32")/255.0
        pred = model.predict(np.expand_dims(img,0), verbose=0)[0]

        # Debug: print raw probabilities
        print(f"Frame {frame_idx}: No-Weapon={pred[0]:.3f}, Weapon={pred[1]:.3f}")

        # Apply threshold logic
        if pred[1] > 0.6:   # weapon probability > 60%
            idx = 1
            conf = pred[1]
        else:
            idx = 0
            conf = pred[0]

        label = f"{class_names[idx]} {conf*100:.1f}%"
        color = (0,255,0) if idx==0 else (0,0,255)

        # Draw on frame
        cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame,(5,5),(w-5,h-5), color, 4)

        # Log to CSV
        writer.writerow([frame_idx, time.time(), class_names[idx], conf, pred[0], pred[1]])
        log.flush()

    out_vid.write(frame)

    cv2.imshow("Weapon Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_idx += 1

cap.release()
out_vid.release()
log.close()
cv2.destroyAllWindows()
