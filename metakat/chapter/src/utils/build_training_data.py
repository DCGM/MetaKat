# Script for building training data for GNN and XGBoost models.
# Author: Richard Blažo
# File name: build_training_data.py
# Description: This script is used to build training data for the XGBoost and GNN models from Label Studio annotations.
# Parts of code were designed with the use of ChatGPT.

from typing import List

import torch
from torch_geometric.data import Data

from src.utils.coordinate_conversions import labelStudioToYOLO


def parse_label(label: List[str]) -> str:
    if label:
        if label[0] == "kapitola":
            return "chapter"
        if label[0] == "jiny nadpis":
            return "subchapter"
    return ""


def build_training_data_xgb(pages):
    trainingData = []
    for page in pages:
        page_chapters = []
        if not page.get("annotations"):
            print(f"Warning: No annotations found on page {page['id']}.")
            continue

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

                    realClass = 0 if label == "chapter" else 1
                    box = {"coords": coords, "classId": realClass}
                    page_chapters.append(box)

        page_chapters.sort(key=lambda x: x["coords"][1])
        if not page_chapters:
            continue
        trainingData.append(page_chapters)

    return trainingData


# Following part of the code was designed with the use of ChatGPT
def build_training_graph_gnn(pages):
    all_graphs = []

    for page in pages:
        nodes = []
        edges = []

        page_items = []

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
                    item = {
                        "type": label,
                        "x": coords[0],
                        "y": coords[1],
                        "width": coords[2],
                        "height": coords[3],
                    }
                    page_items.append(item)

        if not page_items:
            print(f"Warning: No items found on page {page['id']}.")
            continue

        page_items.sort(key=lambda x: x["y"])

        synthetic_node_added = False
        if page_items and page_items[0]["type"] == "subchapter":
            prev_node = {
                "type": "chapter",
                "x": 0,
                "y": 0,
                "width": 0,
                "height": 0,
            }
            nodes.append(prev_node)
            nodes.extend(page_items)
            start = 1
            prevNodeId = 0
            synthetic_node_added = True
        else:
            nodes.extend(page_items)
            start = 0
            prevNodeId = None

        # All nodes will simply be connected to the next one
        for i in range(start, len(nodes) - 1):
            if prevNodeId is not None:
                edges.append((prevNodeId, i))
            else:
                prevNodeId = i
        x = []
        labels = []

        # Features:
        # [left_x, y, relative_x, width, height, aspect_ratio, y_diff, x_diff, is_synthetic]

        if synthetic_node_added:
            x.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
            labels.append(0)
            nodes = nodes[1:]

        left_x_list = [
            x - (w / 2) for (x, w) in [(node["x"], node["width"]) for node in nodes]
        ]

        average_x = sum(left_x_list) / len(nodes) if nodes else 0

        prev_x = None
        prev_y = None

        for node in nodes:
            left_x = node["x"] - (node["width"] / 2)
            labels.append(0 if node["type"] == "chapter" else 1)
            relative_x = left_x - average_x
            aspect_ratio = node["width"] / \
                node["height"] if node["height"] > 0 else 0
            x_diff = left_x - prev_x if prev_x is not None else 0
            y_diff = node["y"] - prev_y if prev_y is not None else 0
            prev_x = left_x
            prev_y = node["y"]

            x.append([left_x, node["y"], relative_x,
                     node["width"], node["height"], aspect_ratio, y_diff, x_diff, 0])

        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        x_tensor = torch.tensor(x, dtype=torch.float)
        y_tensor = torch.tensor(labels, dtype=torch.long)

        edge_labels = []
        edge_label_index = [[], []]

        for src, dst in edges:
            src_label = labels[src]
            dst_label = labels[dst]
            is_hierarchical = int(src_label == 0 and dst_label == 1)

            edge_label_index[0].append(src)
            edge_label_index[1].append(dst)
            edge_labels.append(is_hierarchical)

        edge_label_index = torch.tensor(edge_label_index, dtype=torch.long)
        edge_labels = torch.tensor(edge_labels, dtype=torch.float)

        graph = Data(
            x=x_tensor,
            edge_index=edge_index,
            y=y_tensor,
            edge_label_index=edge_label_index,
            edge_labels=edge_labels,
        )
        all_graphs.append(graph)

    return all_graphs


def boxes_to_graph(yolo_boxes: list, useSyntheticNode: bool) -> Data:
    if not yolo_boxes:
        raise Exception("No boxes provided.")
    sorted_boxes = sorted([box["coords"]
                          for box in yolo_boxes], key=lambda x: x[1])
    left_x_list = [
        x - (w / 2) for (x, w) in [(box[0], box[2]) for box in sorted_boxes]
    ]

    average_x = sum(left_x_list) / len(sorted_boxes) if sorted_boxes else 0

    prev_x = None
    prev_y = None

    new_boxes = []
    edges = []

    # Synthetic node is added if previous box is subchapter
    if useSyntheticNode:
        new_boxes.insert(0, [0, 0, 0, 0, 0, 0, 0, 0, 1])
        edges.insert(0, (0, 1))

    start = 1 if useSyntheticNode else 0

    for i in range(start, len(sorted_boxes) - 1):
        edges.append((i, i + 1))

    for box in sorted_boxes:
        x, y, w, h = box
        left_x = x - (w / 2)
        relative_x = left_x - average_x
        aspect_ratio = w / h if h != 0 else 0
        x_diff = left_x - prev_x if prev_x is not None else 0
        y_diff = y - prev_y if prev_y is not None else 0

        new_boxes.append(
            [left_x, y, relative_x, w, h, aspect_ratio, y_diff, x_diff, 0])

        prev_x = left_x
        prev_y = y

    x_tensor = torch.tensor(new_boxes, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    graph = Data(x=x_tensor, edge_index=edge_index)
    return graph
