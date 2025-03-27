from ultralytics import YOLO
import argparse
import os
import json

modelVersion = "v0.0.3"

classes = ['cislo strany', 'jine cislo', 'jiny nadpis', 'kapitola', 'podnadpis', 'nadpis v textu']

parser = argparse.ArgumentParser(description='Classify images')
parser.add_argument('--source', type=str, default='data/images', help='Source directory')
parser.add_argument('--imgOutput', type=str, default='classify/image-results', help='Annotated images output directory')
parser.add_argument('--txtOutput', type=str, default='classify/box-results', help='Output directory for bounding boxes')
parser.add_argument('--listOutput', type=str, default='classify/found-files.txt', help='Output file for found files')
parser.add_argument('--weights', type=str, default='res/best.pt', help='Path to the weights file')
parser.add_argument('--iou', type=float, default=0.15, help='IoU threshold')
parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold')
args = parser.parse_args()

inputF = args.source
outputImages = args.imgOutput
outputBoxes = args.txtOutput

foundFiles = []

model = YOLO(args.weights)

def classify(image_path):
    results = model(image_path, conf=args.conf, iou=args.iou, agnostic_nms=True, imgsz=1280, verbose=False)

    chapterConfidences = []

    hasChapterInText = False

    for result in results:
        
        filename = os.path.basename(result.path)
        boxes = result.boxes
        predictionCounter = 0
        predictionResults = []
        for i in range(len(boxes.xywhn)):
            boxToAppend = {}
            boxToAppend["classId"] = int(boxes.cls[i].item())
            classIndex = int(boxes.cls[i].item())
            if boxToAppend["classId"] == 5:
                chapterConfidences.append(boxes.conf[i].item())
                hasChapterInText = True
            boxToAppend["coords"] = boxes.xywhn[i].tolist()
            boxToAppend["conf"] = boxes.conf[i].item()
            predictionCounter += 1
            coordinates = boxes.xywhn[i].tolist()
            w = coordinates[2]
            h = coordinates[3]
            x = coordinates[0] - w/2
            y = coordinates[1] - h/2
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
            newPrediction = {
                "id": filename + "." + str(predictionCounter),
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": result.orig_shape[1],
                "original_height": result.orig_shape[0],
                "image_rotation": 0,
                "value": predictionValue,
                "score": boxes.conf[i].item()
            }
            predictionResults.append(newPrediction)
        if not hasChapterInText:
            continue
        jsonOutput = {
            "data": {
                "image": "/data/local-files/?d=digilinka_obsahy/images/" + filename,
            },
            "predictions": [{
                "model_version": modelVersion,
                "result": predictionResults
            }]
        }
        with open(os.path.join(outputBoxes, os.path.splitext(filename)[0] + '.json'), 'w') as outfile:
            json.dump(jsonOutput, outfile, indent=4)
        #Save predictions as well, just for now
        result.save(filename=os.path.join(outputImages, filename))

    if chapterConfidences: 
      print(f"{filename} had a chapter name in text, saving results.")
      foundFiles.append((filename, chapterConfidences))
    return

for file in os.listdir(inputF):
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
        classify(os.path.join(inputF, file))

with open(args.listOutput, 'w') as f:
    for item in foundFiles:
        f.write(f"{item[0]}: {item[1]}\n")
    f.close()

print("Files saved to", args.listOutput)