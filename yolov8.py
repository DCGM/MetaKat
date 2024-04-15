import sys
import os
from ultralytics import YOLO

if len(sys.argv) != 3:
    print('Usage: python yolov8.py <dataset> <project>')
    sys.exit(1)
    
dataset_dir = sys.argv[1]
project_dir = sys.argv[2]

model = YOLO(os.path.join(dataset_dir, 'yolov8n.yaml'))
results = model.train(data=os.path.join(dataset_dir, 'dataset.yaml'),
                      project=project_dir,
                      epochs=2000)


