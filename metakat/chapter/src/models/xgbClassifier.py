# This file with XGBoost training and prediction functions.
# Author: Richard Blažo
# File name: xgbClassifier.py
# Description: File contains functions for training and using an XGBoost classifier model
# for predicting the hierarchy of chapters and subchapters.
import numpy as np
import xgboost as xgb

from src.utils.classes import DetectionClass


def train_chapter_classifier(entries):
    features_list = []
    labels = []

    for page_entries in entries:
        prev_x = prev_y = None
        prev_class = page_entries[0]["classId"] if page_entries else None

        left_x_list = [
            x - (w / 2) for (x, _, w, _) in [entry["coords"] for entry in page_entries]
        ]

        average_x = sum(left_x_list) / len(page_entries) if page_entries else 0

        for entry in page_entries:
            x, y, w, h = entry["coords"]
            left_x = x - (w / 2)
            rel_x = left_x - average_x
            aspect_ratio = w / h if h != 0 else 0

            if prev_x is not None and prev_y is not None:
                x_diff = x - prev_x
                y_diff = y - prev_y
            else:
                x_diff = -2.0
                y_diff = -2.0

            features = [left_x, y, w, h, rel_x,
                        x_diff, y_diff, aspect_ratio, prev_class]
            features_list.append(features)
            labels.append(entry["classId"])

            prev_x, prev_y = left_x, y
            prev_class = entry["classId"]

    X = np.array(features_list)
    y = np.array(labels)

    model = xgb.XGBClassifier(
        n_estimators=2000,
        reg_alpha=0.1,
        reg_lambda=1.0,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.01,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    model.fit(X, y)
    print("Model training complete.")
    return model


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
