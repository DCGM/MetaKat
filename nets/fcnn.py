# File: fcnn.py
# Author: Jakub Křivánek
# Date: 7. 5. 2024
# Description: This file contains the FCNN model and the dataset class for the FCNN model.

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class FCNN(nn.Module):
    def __init__(self, classes, input_keys, neighbour_lines_cnt=0):
        super().__init__()
        self.classes = classes
        self.input_keys = input_keys
        self.input_size = len(input_keys) * (neighbour_lines_cnt * 2 + 1)

        upsampled_size = 128
        self.fc1 = nn.Linear(self.input_size, upsampled_size)
        self.fc2 = nn.Linear(upsampled_size, 64)
        self.fc3 = nn.Linear(64, len(classes))

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class JsonDataset(Dataset):
    def __init__(self, json_file, classes, input_keys, neighbour_lines_cnt=0, train=False, val=False):
        self.neighbour_lines_cnt = neighbour_lines_cnt
        self.classes = classes
        self.input_keys = input_keys

        with open(json_file) as f:
            self.data = json.load(f)

        if train:
            self.ids = self.data["train_ids"]
        elif val:
            self.ids = self.data["val_ids"]
        else:
            self.ids = self.data["train_ids"] + self.data["val_ids"]

    def __len__(self):
        return len(self.ids)

    def get_line_features(self, idx, current_line_idx):
        if 0 <= idx < len(self.data["lines"]) and \
           self.data["lines"][idx]["page_id"] == self.data["lines"][current_line_idx]["page_id"]:
            return [self.data["lines"][idx][key] for key in self.input_keys]
        else:
            return [0] * len(self.input_keys)

    def __getitem__(self, idx):
        labels = self.data["lines"][self.ids[idx]]["labels"]
        labels = torch.tensor([labels.get(label, 0) for label in self.classes], dtype=torch.float32)
        current_line = torch.tensor([self.data["lines"][self.ids[idx]][key] for key in self.input_keys])

        if self.neighbour_lines_cnt > 0:
            neighbor_indices = [
                idx + offset
                for delta in range(1, self.neighbour_lines_cnt + 1)
                for offset in (-delta, delta)
            ]
            neighbor_indices = sorted(neighbor_indices, key=lambda x: x)

            neighbour_lines = []
            for neighbor_idx in neighbor_indices:
                neighbour_line = self.get_line_features(neighbor_idx, idx)
                neighbour_lines.extend(neighbour_line)

            neighbour_lines_tensor = torch.tensor(neighbour_lines)
            input_tensor = torch.cat((current_line, neighbour_lines_tensor))
        else:
            input_tensor = current_line

        return {"input": input_tensor, "target": labels}


class JsonDatasetRenderer(JsonDataset):
    # provisional dataset for rendering images
    def __init__(self, json_file, classes, input_keys, neighbour_lines_cnt=0):
        self.neighbour_lines_cnt = neighbour_lines_cnt
        self.classes = classes
        self.input_keys = input_keys

        with open(json_file) as f:
            self.data = json.load(f)

        self.ids = range(len(self.data["lines"]))

    def __getitem__(self, idx):
        labels = self.data["lines"][self.ids[idx]]["labels"]
        labels = torch.tensor([labels.get(label, 0) for label in self.classes], dtype=torch.float32)
        current_line = torch.tensor([self.data["lines"][self.ids[idx]][key] for key in self.input_keys])

        if self.neighbour_lines_cnt > 0:
            neighbor_indices = [
                idx + offset
                for delta in range(1, self.neighbour_lines_cnt + 1)
                for offset in (-delta, delta)
            ]
            neighbor_indices = sorted(neighbor_indices, key=lambda x: x)

            neighbour_lines = []
            for neighbor_idx in neighbor_indices:
                neighbour_line = self.get_line_features(neighbor_idx, idx)
                neighbour_lines.extend(neighbour_line)

            neighbour_lines_tensor = torch.tensor(neighbour_lines)
            input_tensor = torch.cat((current_line, neighbour_lines_tensor))
        else:
            input_tensor = current_line

        paddings = ["padding_top", "padding_right", "padding_bottom", "padding_left"]
        paddings = [padding + "_not_normalized" for padding in paddings]
        not_normalized_padding = torch.tensor([self.data["lines"][self.ids[idx]][key] for key in paddings])
        return {"input": input_tensor, "target": labels, "padding": not_normalized_padding, "page_id": self.data["lines"][self.ids[idx]]["page_id"]}
