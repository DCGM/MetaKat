import os
import argparse
import yaml
import json
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Classify images')
parser.add_argument('--source', type=str, default='classify/nkp/images', help='Source directory')
parser.add_argument('--imgOutput', type=str, default='classify/results', help='Annotated images output directory')
parser.add_argument('--txtOutput', type=str, default='classify/nkp/boxes', help='Output directory for bounding boxes')
parser.add_argument('--weights', type=str, default='res/weights/best.pt', help='Path to the weights file')
parser.add_argument('--dataConfig', type=str, default='dataset-config.yaml', help='Path to dataset config file')
parser.add_argument('--iou', type=float, default=0.3, help='IoU threshold')
args = parser.parse_args()

modelVersion = "v0.0.1"

### TODO: Cleanup and split into multiple functions.

with open(args.dataConfig) as file:
    data = yaml.safe_load(file)

classes = data['names']
print(f"Classes: {classes}")

model = YOLO(args.weights)
model.overrides['iou'] = args.iou

inputF = args.source
outputF = args.imgOutput
outputBoxes = args.txtOutput

os.makedirs(outputF, exist_ok=True)


for file in os.listdir(inputF):
    if file.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(inputF, file)
        
        results = model(image_path)
        
        # Save the output to the specified folder
        for result in results:
            filename = os.path.basename(result.path)
            origshape = result.orig_shape
            boxes = result.boxes
            masks = result.masks
            keypoints = result.keypoints
            probs = result.probs
            obb = result.obb
            predictionCounter = 0
            predictionResults = []
            for i in range(len(boxes.xywhn)):
                predictionCounter += 1
                classIndex = int(boxes.cls[i].item())
                predictionId = filename + "." + classes[classIndex] + str(predictionCounter)
                predictionOriginalWidth = origshape[1]
                predictionOriginalHeight = origshape[0]
                coordinates = boxes.xywhn[i].tolist()
                x = coordinates[0]
                y = coordinates[1]
                w = coordinates[2]
                h = coordinates[3]
                predictionValue = {
                    "rotation": 0,
                    "x": x*100,
                    "y": y*100,
                    "width": w*100,
                    "height": h*100,
                    "rectanglelabels": [
                        classes[classIndex]
                    ],
                }
                predictionScore = boxes.conf[i].item()
                newPrediction = {
                    "id": predictionId,
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": predictionOriginalWidth,
                    "original_height": predictionOriginalHeight,
                    "image_rotation": 0,
                    "value": predictionValue,
                    "score": predictionScore
                }
                predictionResults.append(newPrediction)
            jsonOutput = {
                "data": {
                    "image": "/data/local-files/?d=digilinka/images/" + filename,
                },
                "predictions": [{
                    "model_version": modelVersion,
                    "results": predictionResults
                }]
            }
            with open(os.path.join(outputBoxes, filename + '.json'), 'w') as outfile:
                json.dump(jsonOutput, outfile, indent=4)
            #Save predictions as well, just for now
            result.save(filename=os.path.join(outputF, filename))
        print(f"Processed {file}")

