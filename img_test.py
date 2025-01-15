from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Increase file size limit
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB


# Load YOLOv8 model
model = YOLO(r'D:\AI-20250112T101556Z-001\AI\results\experiment_fast\weights\best.pt')

# Folder to store annotated images/videos
OUTPUT_FOLDER = 'static/output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return "Card Detection API"

@app.route('/predict-image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Read the uploaded image
    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform object detection
    results = model(image)
    detections = []

    # Save annotated image
    annotated_image = results[0].plot()  # YOLOv8 method to draw detections on the image
    output_path = os.path.join(OUTPUT_FOLDER, 'annotated_image.jpg')
    cv2.imwrite(output_path, annotated_image)

    # Prepare detection data
    for detection in results[0].boxes:
        box = detection.xyxy[0].tolist()  # Bounding box coordinates
        conf = float(detection.conf[0])  # Confidence score
        cls = int(detection.cls[0])  # Class ID

        detections.append({
            'bbox': [int(coord) for coord in box],
            'confidence': conf,
            'class': cls
        })

    return jsonify({
        'detections': detections,
        'output_image_url': f'http://127.0.0.1:5000/{output_path}'
    })

@app.route('/predict-video', methods=['POST'])
def predict_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    # Read the uploaded video
    video_file = request.files['video']
    video_path = os.path.join(OUTPUT_FOLDER, 'uploaded_video.mp4')
    video_file.save(video_path)

    # Open video for processing
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Output video path
    output_video_path = os.path.join(OUTPUT_FOLDER, 'annotated_video.mp4')
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on each frame
        results = model(frame)
        annotated_frame = results[0].plot()  # Draw detections on the frame
        out.write(annotated_frame)

    cap.release()
    out.release()

    return jsonify({
        'output_video_url': f'http://127.0.0.1:5000/{output_video_path}'
    })

@app.route('/static/output/<path:filename>')
def serve_image_or_video(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True)
