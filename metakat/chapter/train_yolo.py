# YOLOv8 training script
# Author: Richard Blažo
# File name: train_yolo.py
# Description: Script used to train a YOLOv8 model for object detection.


import os

os.environ["ULTRALYTICS_CONFIG_DIR"] = os.getcwd()
from ultralytics import YOLO

model = YOLO('yolov8s.yaml')

model.train(
    data='new.yaml',
    epochs=500,
    patience=150,
    imgsz=(960, 1280),
    batch=16,
    name='BP',
    project='results',
    close_mosaic=10,
    augment=True
)
