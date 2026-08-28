# Parcel damage detection - webcam inference

import cv2
import os
import sys

try:
    from ultralytics import YOLO
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")
CONFIDENCE = 0.5

if not os.path.exists(MODEL_PATH):
    print(f"Error: model not found at {MODEL_PATH}")
    sys.exit(1)

model = YOLO(MODEL_PATH)
print(f"Model loaded | Classes: {model.names}")

# Webcam
cap = cv2.VideoCapture(0)
window_name = "Parcel Damage Detector"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

print("\nPress Q or ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    results = model.predict(frame, conf=CONFIDENCE, verbose=False)
    annotated = results[0].plot()

    detections = len(results[0].boxes)
    cv2.putText(annotated, f"Detections: {detections}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(window_name, annotated)

    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
