from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load a pre-trained model
results = model.predict(source="dte_plycrd/Images/Images/2C0.jpg")  # Predict using an image
print(results);