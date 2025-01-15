from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from ultralytics import YOLO
from flask_cors import CORS
import base64


app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable WebSocket communication

# Load YOLOv8 model
model = YOLO(r'D:\project\playing-card-detection\results\experiment_fast\weights\best.pt')  # Update with your model's path

# Map class IDs to card values (example for Blackjack)
CARD_VALUES = {
    0: ('10', 10), 1: ('10', 10), 2: ('10', 10), 3: ('10', 10),
    4: ('2', 2), 5: ('2', 2), 6: ('2', 2), 7: ('2', 2),
    8: ('3', 3), 9: ('3', 3), 10: ('3', 3), 11: ('3', 3),
    12: ('4', 4), 13: ('4', 4), 14: ('4', 4), 15: ('4', 4),
    16: ('5', 5), 17: ('5', 5), 18: ('5', 5), 19: ('5', 5),
    20: ('6', 6), 21: ('6', 6), 22: ('6', 6), 23: ('6', 6),
    24: ('7', 7), 25: ('7', 7), 26: ('7', 7), 27: ('7', 7),
    28: ('8', 8), 29: ('8', 8), 30: ('8', 8), 31: ('8', 8),
    32: ('9', 9), 33: ('9', 9), 34: ('9', 9), 35: ('9', 9),
    36: ('Ace', (1, 11)), 37: ('Ace', (1, 11)), 38: ('Ace', (1, 11)), 39: ('Ace', (1, 11)),
    40: ('Jack', 10), 41: ('Jack', 10), 42: ('Jack', 10), 43: ('Jack', 10),
    44: ('King', 10), 45: ('King', 10), 46: ('King', 10), 47: ('King', 10),
    48: ('Queen', 10), 49: ('Queen', 10), 50: ('Queen', 10), 51: ('Queen', 10)
}


@app.route('/')
def home():
    return render_template('dsh.html')  # Frontend to display video feed and results

@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Handle real-time video frames sent via WebSocket.
    """
    # Decode base64 frame
    frame_data = data.split(',')[1]  # Remove the base64 header part
    frame_bytes = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
    frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)

    # Predict with YOLOv8
    detections = model(frame)
    frame_results = []

    # Process YOLO detections
    for detection in detections[0].boxes:  # Access first result
        box = detection.xyxy[0].tolist()  # [x1, y1, x2, y2]
        conf = float(detection.conf[0])  # Confidence
        cls = int(detection.cls[0])  # Class ID

        card_name, card_value = CARD_VALUES.get(cls, ('Unknown', 0))
        frame_results.append({
            'bbox': [int(coord) for coord in box],
            'confidence': conf,
            'class': cls,
            'card': card_name,
            'value': card_value
        })

    # Analyze card values for Blackjack
    total_value = 0
    ace_count = 0
    for result in frame_results:
        value = result['value']
        if isinstance(value, tuple):  # Ace can be 1 or 11
            ace_count += 1
            total_value += value[1]  # Default to 11
        else:
            total_value += value

    # Adjust for Aces if total exceeds 21
    while total_value > 21 and ace_count > 0:
        total_value -= 10
        ace_count -= 1

    # Send predictions back to the frontend
    emit('predictions', {'frame_results': frame_results, 'total_value': total_value})

@socketio.on('connect')
def on_connect():
    print("Client connected!")

@socketio.on('disconnect')
def on_disconnect():
    print("Client disconnected!")

if __name__ == '__main__':
    socketio.run(app, debug=True)
