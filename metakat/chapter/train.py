from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='dataset-config.yaml',
    epochs=10,
    imgsz=640,
    batch=4,
    name='BP',  # Save folder for results
    project='results'  # Path to save training results
)