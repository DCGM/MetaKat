import os
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
        
    return parser.parse_args()

args = parse_args()
    
dataset_dir = args.dataset_dir
project_dir = args.output_dir
config_file = os.path.join(dataset_dir, 'config.yaml')

model = YOLO(os.path.join(dataset_dir, 'yolov8n.yaml'))

if os.path.exists(config_file):
    model.train(data=os.path.join(dataset_dir, 'dataset.yaml'),
                      project=project_dir,
                      cfg=config_file,
                      epochs=args.epochs)
else:
    model.train(data=os.path.join(dataset_dir, 'dataset.yaml'),
                      project=project_dir,
                      epochs=args.epochs)


