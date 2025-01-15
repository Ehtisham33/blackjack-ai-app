from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)  # Enable CORS if needed

# Load YOLOv8 model
model = YOLO(r'D:\AI-20250112T101556Z-001\AI\results\experiment_fast\weights\best.pt')  # Update with your model's path

@app.route('/')
def home():
    return "YOLOv8 Flask Server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    # Save the uploaded video temporarily
    video_file = request.files['video']
    video_path = 'uploaded_video.mp4'
    video_file.save(video_path)

    # Read and process the video
    cap = cv2.VideoCapture(video_path)
    results = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Predict with YOLOv8
        detections = model(frame)  # Frame predictions
        frame_results = []

        # Extract bounding boxes, confidences, and class IDs
        for detection in detections[0].boxes:  # Access first result
            box = detection.xyxy[0].tolist()  # [x1, y1, x2, y2]
            conf = float(detection.conf[0])  # Confidence
            cls = int(detection.cls[0])  # Class ID

            frame_results.append({
                'bbox': [int(coord) for coord in box],
                'confidence': conf,
                'class': cls
            })

        results.append({
            'frame': frame_count,
            'detections': frame_results
        })
        frame_count += 1

    cap.release()
    return jsonify({'predictions': results})

if __name__ == '__main__':
    app.run(debug=True)
