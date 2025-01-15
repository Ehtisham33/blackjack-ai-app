from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from collections import Counter

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load YOLOv8 model
model = YOLO(r'D:\AI-20250112T101556Z-001\AI\results\experiment_fast\weights\best.pt')

# Card Values
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

# Deck Statistics
TOTAL_CARDS_IN_DECK = Counter({
    'Ace': 4, '2': 4, '3': 4, '4': 4, '5': 4, '6': 4, '7': 4,
    '8': 4, '9': 4, '10': 4, 'Jack': 4, 'Queen': 4, 'King': 4
})
played_cards = Counter()


def calculate_card_statistics(played_cards):
    stats = []
    for card, total in TOTAL_CARDS_IN_DECK.items():
        played = played_cards[card]
        played_percentage = (played / total) * 100 if total > 0 else 0
        stats.append({
            'Card Type': card,
            'Played': played,
            'Total in Deck': total,
            'Played %': round(played_percentage, 2)
        })
    return stats


@app.route('/')
def home():
    return render_template('dsh.html')


@socketio.on('video_frame')
def handle_video_frame(data):
    frame_data = data.split(',')[1]
    frame_bytes = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
    frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)

    detections = model(frame)
    frame_results = []

    for detection in detections[0].boxes:
        box = detection.xyxy[0].tolist()
        conf = float(detection.conf[0])
        cls = int(detection.cls[0])

        card_name, card_value = CARD_VALUES.get(cls, ('Unknown', 0))
        if card_name != 'Unknown':
            played_cards[card_name] += 1

        frame_results.append({
            'bbox': [int(coord) for coord in box],
            'confidence': conf,
            'class': cls,
            'card': card_name,
            'value': card_value
        })

    total_value = 0
    ace_count = 0
    for result in frame_results:
        value = result['value']
        if isinstance(value, tuple):
            ace_count += 1
            total_value += value[1]
        else:
            total_value += value

    while total_value > 21 and ace_count > 0:
        total_value -= 10
        ace_count -= 1

    card_statistics = calculate_card_statistics(played_cards)

    emit('predictions', {
        'frame_results': frame_results,
        'total_value': total_value,
        'card_statistics': card_statistics
    })


@socketio.on('connect')
def on_connect():
    print("Client connected!")


@socketio.on('disconnect')
def on_disconnect():
    print("Client disconnected!")


if __name__ == '__main__':
    socketio.run(app, debug=True)
