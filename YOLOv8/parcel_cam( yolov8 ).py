# ============================================================
# 包裹损坏检测 - 推理脚本
# 使用方法: python detect.py
# ============================================================

import cv2
import os
import sys

# 自动安装依赖
try:
    from ultralytics import YOLO
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO

# ==================== 配置 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")
CONFIDENCE = 0.5  # 置信度阈值

# ==================== 加载模型 ====================
if not os.path.exists(MODEL_PATH):
    print(f"错误: 找不到模型文件 {MODEL_PATH}")
    print("请先把 best.pt 放到同目录下")
    sys.exit(1)

model = YOLO(MODEL_PATH)
print(f"模型加载成功!")
print(f"类别: {model.names}")

# ==================== 摄像头检测 ====================
cap = cv2.VideoCapture(0)
window_name = "Parcel Damage Detector"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

print("\n按 Q 或 ESC 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # 推理
    results = model.predict(frame, conf=CONFIDENCE, verbose=False)

    # 画结果
    annotated = results[0].plot()

    # 显示检测数量
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
