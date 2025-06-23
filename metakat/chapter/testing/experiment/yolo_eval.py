# This file contains an experiment for evaluating the YOLO model on a dataset.
# Author: Richard Blažo
# File name: yolo_eval.py
# Description: File contains an experiment for evaluating the YOLO model on a dataset.

import json
import os
from collections import Counter, defaultdict

from ultralytics import YOLO
from src.utils.coordinate_conversions import labelStudioToYOLO


def build_eval_data_YOLO(pages):
    trainingData = []
    counter = Counter()
    for page in pages:
        page_chapters = []
        if not page.get("annotations"):
            print(f"Warning: No annotations found on page {page['id']}.")
            continue

        for result in page.get("annotations", [])[0]["result"]:
            if result["type"] == "rectanglelabels":
                label = result["value"]["rectanglelabels"][0]
                coords = labelStudioToYOLO(
                    (
                        result["value"]["x"],
                        result["value"]["y"],
                        result["value"]["width"],
                        result["value"]["height"],
                    )
                )

                if label == "kapitola":
                    real_class = 3
                elif label == "jiny nadpis":
                    real_class = 2
                elif label == "nadpis v textu":
                    real_class = 4
                elif label == "podnadpis":
                    real_class = 5
                elif label == "cislo strany":
                    real_class = 0
                elif label == "jine cislo":
                    real_class = 1
                else:
                    print(
                        f"Warning: Unknown label {label} on page {page['id']}.")
                    continue
                counter[real_class] += 1
                box = {"coords": coords, "classId": real_class}
                page_chapters.append(box)

        page_chapters.sort(key=lambda x: x["coords"][1])
        if not page_chapters:
            continue
        trainingData.append(
            {"filename": page["data"]["image"], "boxes": page_chapters})
    return trainingData


def load_yolo_predictions(yolo_output_path):
    predictions = []
    if not os.path.exists(yolo_output_path):
        print(f"Warning: YOLO output file {yolo_output_path} does not exist.")
        return predictions
    for line in open(yolo_output_path).readlines():
        parts = line.strip().split()
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        predictions.append({
            "coords": (x, y, w, h),
            "classId": class_id
        })
    return predictions


def calculate_iou(box1, box2):
    def get_corner(box):
        x, y, w, h = box
        return (x - w/2, y - h/2, x + w/2, y + h/2)

    x1_min, y1_min, x1_max, y1_max = get_corner(box1)
    x2_min, y2_min, x2_max, y2_max = get_corner(box2)

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * \
        max(0, inter_ymax - inter_ymin)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area != 0 else 0


model = YOLO(
    "models/YOLO/text_sep.pt")
results = model.predict(
    source="data/YOLO/TEST/TEXT/images/", save=False, conf=0.25, stream=True,)

output_dir = "here"
os.makedirs(output_dir, exist_ok=True)

for result in results:
    image_path = result.path
    image_id = os.path.splitext(os.path.basename(image_path))[
        0]
    output_file = os.path.join(output_dir, f"{image_id}.txt")

    with open(output_file, "w") as f:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x, y, w, h = box.xywh[0].tolist()
            x /= result.orig_shape[1]
            y /= result.orig_shape[0]
            w /= result.orig_shape[1]
            h /= result.orig_shape[0]
            f.write(f"{cls_id} {x} {y} {w} {h}\n")


IOU_THRESHOLD = 0.5
yolo_y_true = []
yolo_y_pred = []

with open("data/YOLO/TEST/TEXT/TEXT.json", "r") as f:
    new_raw_data = json.load(f)

test_data = build_eval_data_YOLO(new_raw_data)

tp_counter = defaultdict(int)
fp_counter = defaultdict(int)
fn_counter = defaultdict(int)

for page_gt in test_data:
    filename = os.path.basename(page_gt["filename"])
    boxes_gt = page_gt["boxes"]

    yolo_preds = load_yolo_predictions(
        f"here/{os.path.splitext(filename)[0]}.txt")
    matched_preds = set()

    for gt_box in boxes_gt:
        matched = False
        for idx, pred_box in enumerate(yolo_preds):
            if idx in matched_preds:
                continue
            if calculate_iou(gt_box["coords"], pred_box["coords"]) >= IOU_THRESHOLD:
                matched_preds.add(idx)
                gt_class = gt_box["classId"]
                pred_class = pred_box["classId"]
                if gt_class == pred_class:
                    tp_counter[gt_class] += 1
                else:
                    fp_counter[pred_class] += 1
                    fn_counter[gt_class] += 1
                matched = True
                break
        if not matched:
            fn_counter[gt_box["classId"]] += 1
            yolo_y_true.append(gt_box["classId"])
            yolo_y_pred.append(-1)

class_ids = sorted(model.names.keys())
for class_id in class_ids:
    tp = tp_counter[class_id]
    fp = fp_counter[class_id]
    fn = fn_counter[class_id]

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (
        precision + recall) > 0 else 0

    print(f"Class {class_id}: TP={tp}, FP={fp}, FN={fn}, Precision={precision:.4f}, \
          Recall={recall:.4f}, F1-Score={f1_score:.4f}")
