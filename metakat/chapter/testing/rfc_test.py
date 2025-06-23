# RFC test script
# Author: Richard Blažo
# File name: rfc_test.py
# Description: Script used to evaluate the Random Forest Classifier model, which
# predicts the page number for a given chapter.

import os
import pickle

import numpy as np

from src.models.random_forest_module import (build_training_dataframe,
                                             compute_features,
                                             parse_labelstudio_annotations,
                                             predict_page_for_chapters,
                                             prepare_forest_classifier)


def get_real_mapping(relations_final):

    true_mapping = {}
    for _, relations in relations_final.items():
        for from_id, to_id in relations:
            true_mapping[to_id] = from_id
    return true_mapping


def evaluate_structured_predictions(model, boxes_final_test, relations_final_test):
    tp = 0
    fp = 0
    fn = 0

    for page_id, bboxes in boxes_final_test.items():
        chapter_bboxes = [box for box in bboxes if box["label"] == "kapitola"]
        page_bboxes = [box for box in bboxes if box["label"] == "cislo strany"]

        for box in chapter_bboxes + page_bboxes:
            box["coords"] = (box["x_center"], box["y_center"],
                             box["w"], box["h"])

        # Simulate situation from inference
        chapter_index_to_id = {i: box["id"]
                               for i, box in enumerate(chapter_bboxes)}

        predicted_raw, _ = predict_page_for_chapters(
            model, list(enumerate(chapter_bboxes)), list(enumerate(page_bboxes)))

        predicted = {
            chapter_index_to_id[ch_i]: page_box if page_box is None else page_box[1]
            for ch_i, page_box in predicted_raw.items()
        }
        true_map = get_real_mapping(
            {page_id: relations_final_test[page_id] if page_id in relations_final_test else []})

        for ch_box in chapter_bboxes:
            ch_id = ch_box["id"]
            predicted_pg = predicted.get(ch_id)
            actual_pg_id = true_map.get(ch_id)

            if predicted_pg is None:
                if ch_id in true_map:
                    fn += 1
                continue

            predicted_pg_id = predicted_pg["id"] if isinstance(
                predicted_pg, dict) else predicted_pg.get("id")

            if predicted_pg_id == actual_pg_id:
                tp += 1
            else:
                fp += 1
                if actual_pg_id:
                    fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) else 0

    print("\nStructured Matching Evaluation:")
    print(f"  True Positives:   {tp}")
    print(f"  False Positives:  {fp}")
    print(f"  False Negatives:  {fn}")
    print(f"  Precision:        {precision:.4f}")
    print(f"  Recall:           {recall:.4f}")
    print(f"  F1 Score:         {f1:.4f}")


def evaluate_on_test_set(model, test_json_path):
    boxes_final_test, relations_final_test = parse_labelstudio_annotations(
        test_json_path)
    df_test = build_training_dataframe(boxes_final_test, relations_final_test)
    df_test["features"] = df_test.apply(compute_features, axis=1)

    X_test = np.vstack(df_test["features"])
    y_test = df_test["label"].values

    y_pred = model.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    # First try to load from a saved model
    model_path = "models/RFC/rfc_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Model loaded from", model_path)
    else:
        print("Model not found, training a new one.")
        model = prepare_forest_classifier()
    test_json_path = "data/YOLO/TEST/TOC/TOC2.json"
    boxes_final_test, relations_final_test = parse_labelstudio_annotations(
        test_json_path)

    evaluate_on_test_set(model, test_json_path)

    print("This will take a second...")
    evaluate_structured_predictions(
        model, boxes_final_test, relations_final_test)
