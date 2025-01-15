import cv2
from ultralytics import YOLO

model = YOLO(r'D:\AI-20250112T101556Z-001\AI\results\experiment_fast\weights\best.pt')
video_path = r"uploaded_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()
class_names = model.names
target_width = 1280
target_height = 720

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (target_width, target_height))
    results = model(frame)  
    detections = results[0].boxes  
    for detection in detections:
        box = detection.xyxy[0].tolist() 
        conf = float(detection.conf[0])  
        cls = int(detection.cls[0])  
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[cls]}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('YOLOv8 Real-Time Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
