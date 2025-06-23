# XGB test script
# Author: Richard Blažo
# File name: xgb_test.py
# Description: Script used to evaluate the XGBoost model, which predicts the hierarchy
# of chapters and subchapters in a book.

import json

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.build_training_data import build_training_data_xgb
from src.utils.classes import DetectionClass


def predict_page_classes_xgb(model, chapters, previous_class):
    predictions = []

    if not chapters:
        return predictions

    chapters.sort(key=lambda x: x["coords"][1])

    left_x_list = [
        x - (w / 2) for (x, _, w, _) in [chapter["coords"] for chapter in chapters]
    ]

    average_x = sum(left_x_list) / len(chapters)
    prev_class = 0 if previous_class is None else previous_class
    prev_x = prev_y = None

    for chapter in chapters:
        x, y, w, h = chapter["coords"]
        left_x = x - (w / 2)
        rel_x = left_x - average_x
        aspect_ratio = w / h if h != 0 else 0
        if prev_x is not None and prev_y is not None:
            xDiff = x - prev_x
            yDiff = y - prev_y
        else:
            xDiff = -2.0
            yDiff = -2.0

        features = np.array([left_x, y, w, h, rel_x, xDiff,
                            yDiff, aspect_ratio, prev_class])
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        print(f"Prediction: {prediction}, Features: {features}")

        predictions.append(
            DetectionClass.CHAPTER if prediction == 0 else DetectionClass.SUBCHAPTER
        )

        prev_class = prediction
        prev_x, prev_y = left_x, y

    return predictions


model = xgb.XGBClassifier()
model.load_model(
    "/home/richard/BPN/bpactual/metakat/chapter/models/XGB/latest.ubj")

with open("data/YOLO/TEST/TOC/TOC.json", "r") as f:
    new_raw_data = json.load(f)

test_data = build_training_data_xgb(new_raw_data)


y_true = []
y_pred = []

for page in test_data:
    # Find first chapter or subchapter on page
    if not page:
        continue
    # Sort by y coordinate
    page.sort(key=lambda x: x["coords"][1])

    previous_class = page[0]["classId"] if page[0]["classId"] else None
    preds = predict_page_classes_xgb(model, page, previous_class)

    pred_labels = [0 if p == DetectionClass.CHAPTER else 1 for p in preds]
    true_labels = [entry["classId"] for entry in page]

    y_pred.extend(pred_labels)
    y_true.extend(true_labels)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred,
      target_names=["chapter", "subchapter"]))
