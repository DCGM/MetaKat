# File containing Random Forest classifier functions.
# Author: Richard Blažo
# File name: random_forest_module.py
# Description: This file contains functions for training, using and creating data for a Random Forest
# Classifier model.
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier

from src.utils.coordinate_conversions import labelStudioToYOLO
from src.utils.utils import debugprint


def parse_labelstudio_annotations(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes_final = {}
    relations_final = {}

    for page in data:
        page_id = page["id"]
        annotations = page["annotations"][0]
        results = annotations["result"]

        bboxes = {}
        relations = []

        for item in results:
            if (
                item["type"] == "rectanglelabels"
                and len(item["value"]["rectanglelabels"]) > 0
                and (
                    item["value"]["rectanglelabels"][0] == "kapitola"
                    or item["value"]["rectanglelabels"][0] == "cislo strany"
                )
            ):
                rect_id = item["id"]
                label_name = item["value"]["rectanglelabels"][0]
                x = item["value"]["x"]
                y = item["value"]["y"]
                w = item["value"]["width"]
                h = item["value"]["height"]

                x, y, w, h = labelStudioToYOLO((x, y, w, h))
                bboxes[rect_id] = {
                    "id": f"{page_id}_{rect_id}",
                    "label": label_name,
                    "x_center": x,
                    "y_center": y,
                    "w": w,
                    "h": h,
                }

            # Relation annotation
            elif item["type"] == "relation":
                fid = item["from_id"]
                tid = item["to_id"]
                from_id = f"{page_id}_{fid}"
                to_id = f"{page_id}_{tid}"
                relations.append((from_id, to_id))

        sorted_bboxes = sorted(
            bboxes.values(), key=lambda box: box["y_center"])
        boxes_final[page_id] = sorted_bboxes
        relations_final[page_id] = relations

    return (boxes_final, relations_final)


def compute_features(row):
    dx = abs(row["chapter_x"] - row["page_x"])
    dy = abs(row["chapter_y"] - row["page_y"])

    dw_ratio = row["chapter_w"] / (row["page_w"] + 1e-6)
    dh_ratio = row["chapter_h"] / (row["page_h"] + 1e-6)

    ch_y1, ch_y2 = row["chapter_y"], row["chapter_y"] + row["chapter_h"]
    pg_y1, pg_y2 = row["page_y"], row["page_y"] + row["page_h"]

    y_overlap = max(0, min(ch_y2, pg_y2) - max(ch_y1, pg_y1))

    page_before_chapter = row["page_x"] < row["chapter_x"]

    return np.array([dx, dy, dw_ratio, dh_ratio, y_overlap, page_before_chapter])


def build_training_dataframe(boxes_final, relations_final):
    all_rows = []

    for page_id, bboxes_list in boxes_final.items():
        chapter_bboxes = [
            box for box in bboxes_list if box["label"] == "kapitola"]
        page_bboxes = [
            box for box in bboxes_list if box["label"] == "cislo strany"]

        page_relations = set(relations_final[page_id])
        if len(chapter_bboxes) == 0 or len(page_bboxes) == 0:
            continue

        for ch_box in chapter_bboxes:
            for pg_box in page_bboxes:
                is_valid = (pg_box["id"], ch_box["id"]) in page_relations
                label_val = 1 if is_valid else 0

                row = {
                    "chapter_x": ch_box["x_center"],
                    "chapter_y": ch_box["y_center"],
                    "chapter_w": ch_box["w"],
                    "chapter_h": ch_box["h"],
                    "page_x": pg_box["x_center"],
                    "page_y": pg_box["y_center"],
                    "page_w": pg_box["w"],
                    "page_h": pg_box["h"],
                    "label": label_val,
                }
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    return df


def prepare_forest_classifier(json_target=None):
    debugprint("Preparing model to predict page numbers to chapter names")

    json_file = json_target or "./data/RFC/tasks_updated.json"
    boxes_final, relations_final = parse_labelstudio_annotations(json_file)

    df = build_training_dataframe(boxes_final, relations_final)

    df["features"] = df.apply(compute_features, axis=1)

    X = np.vstack(df["features"])
    y = df["label"].values

    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )

    print("Training model...")
    model.fit(X, y)
    print("Model training completed.")

    return model


def predict_page_for_chapters(model, chapter_bboxes, page_bboxes):
    num_chapters = len(chapter_bboxes)
    num_pages = len(page_bboxes)

    score_matrix = np.zeros((num_chapters, num_pages))

    for ch_i, ch_box in chapter_bboxes:
        for p_i, pg_box in page_bboxes:
            row = {
                "chapter_x": ch_box["coords"][0],
                "chapter_y": ch_box["coords"][1],
                "chapter_w": ch_box["coords"][2],
                "chapter_h": ch_box["coords"][3],
                "page_x": pg_box["coords"][0],
                "page_y": pg_box["coords"][1],
                "page_w": pg_box["coords"][2],
                "page_h": pg_box["coords"][3],
            }

            features = compute_features(row)
            score = model.predict_proba([features])[0][1]

            cost = 1.0 - score
            score_matrix[ch_i][p_i] = cost

    row_ind, col_ind = linear_sum_assignment(score_matrix)

    predicted_pages = {}
    used_page_indexes = set()

    for ch_i, pg_i in zip(row_ind, col_ind):
        cost = score_matrix[ch_i][pg_i]
        score = 1.0 - cost

        if score >= 0.65:
            predicted_pages[ch_i] = (pg_i, page_bboxes[pg_i][1])
            used_page_indexes.add(pg_i)
        else:
            predicted_pages[ch_i] = None

    free_page_numbers = []
    for pg_i, pg_box in page_bboxes:
        if pg_i not in used_page_indexes:
            free_page_numbers.append(pg_box)

    return predicted_pages, free_page_numbers
