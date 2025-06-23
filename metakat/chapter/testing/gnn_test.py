# GNN test script
# Author: Richard Blažo
# File name: gnn_test.py
# Description: Script used to evaluate the GNN model for chapter hierarchy classification.
# Parts of code were designed with the use of ChatGPT.

import json

import torch
from sklearn.metrics import classification_report

from src.utils.build_training_data import boxes_to_graph, parse_label
from src.utils.coordinate_conversions import labelStudioToYOLO
from src.models.gnn_module import MultiTaskGNN

# This section was designed with the use of ChatGPT.


def predict_page_classes_gnn(model, yolo_boxes, device, use_synthetic=True):
    model.eval()
    with torch.no_grad():
        graph = boxes_to_graph(yolo_boxes, use_synthetic).to(device)
        node_logits, _ = model(graph.x, graph.edge_index)
        preds = node_logits.argmax(dim=1).cpu().tolist()
        return [0 if p == 0 else 1 for p in preds]


def load_boxes_from_page(page):
    boxes = []
    for result in page.get("annotations", [])[0]["result"]:
        if result["type"] == "rectanglelabels":
            label = parse_label(result["value"]["rectanglelabels"])
            if label in ["chapter", "subchapter"]:
                coords = labelStudioToYOLO(
                    (
                        result["value"]["x"],
                        result["value"]["y"],
                        result["value"]["width"],
                        result["value"]["height"],
                    )
                )
                boxes.append(
                    {"coords": coords, "classId": 0 if label == "chapter" else 1})
    return boxes


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskGNN().to(device)
    model.load_state_dict(torch.load(
        "models/GNN/chapter_classifier_gnn.pth", weights_only=True))
    model.eval()

    with open("data/YOLO/TEST/BOOKS/BOOKS.json", "r") as f:
        pages = json.load(f)

    true_labels_all = []
    pred_labels_all = []

    for page in pages:
        boxes = load_boxes_from_page(page)
        if not boxes:
            continue

        yolo_boxes = [b for b in boxes]
        true_labels = [b["classId"] for b in boxes]

        pred_labels = predict_page_classes_gnn(
            model, yolo_boxes, device, use_synthetic=False)

        true_labels_all.extend(true_labels)
        pred_labels_all.extend(pred_labels)

    print(classification_report(true_labels_all, pred_labels_all,
          target_names=["chapter", "subchapter"]))
