# YOLO test script
# Author: Richard Blažo
# File name: yolo_test.py
# Description: Script used to evaluate the YOLOv8 model for object detection.


import argparse

from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Test YOLO model")
parser.add_argument("--weights", type=str,
                    default="models/YOLO/gen.pt", help="YOLO weights file")
parser.add_argument("--yaml", type=str,
                    default="dataset-config.yaml", help="YOLO dataset yaml file")

args = parser.parse_args()

model = YOLO(args.weights)
metrics1 = model.val(data=args.yaml, conf=0.001, iou=0.7, imgsz=960,
                     save_json=True, save_conf=True, plots=True, batch=4)
print("F1 per class:", metrics1.box.f1)


metrics2 = model.val(data=args.yaml, conf=0.35, iou=0.15, imgsz=960,
                     save_json=True, save_conf=True, plots=True, batch=4)
print("F1 per class:", metrics2.box.f1)
