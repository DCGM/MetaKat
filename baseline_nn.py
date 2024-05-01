import argparse
import os
import json
import numpy as np
import math
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--net", choices=["baseline", "transformer"], required=True)
    parser.add_argument("--dataset", required=True)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.00001)
    parser.add_argument("--decay-start", type=int, default=1000)
    parser.add_argument("--decay-rate", type=float, default=0.0)
    parser.add_argument("--decay-step", type=int, default=0)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--input-keys", nargs="+", required=True)
    parser.add_argument("--neighbour-lines-cnt", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model-path", required=True)
    parser.add_argument('--positional-encoding', choices=['2d', '1d-page', '1d-seq'], default='2d', help='Possitional encoding for the model')
    parser.add_argument('--positional-encoding-max-len', type=int, default=1000, help='Max len for positional encoding')

    parser.add_argument("--render-val-images", action="store_true")
    parser.add_argument("--mastercopy-dir")
    parser.add_argument("--output-render-dir")

    args = parser.parse_args()

    args.classes = args.classes.split(";")
    return args


def map_label(classes, label):
    return classes.index(label)


class BaselineNet(nn.Module):
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
        current_line_idx = self.data["lines"].index(self.data["lines"][self.ids[idx]])

        if self.neighbour_lines_cnt > 0:
            neighbor_indices = [
                current_line_idx + offset
                for delta in range(1, self.neighbour_lines_cnt + 1)
                for offset in (-delta, delta)
            ]
            neighbor_indices = sorted(neighbor_indices, key=lambda x: x)

            neighbour_lines = []
            for neighbor_idx in neighbor_indices:
                neighbour_line = self.get_line_features(neighbor_idx, current_line_idx)
                neighbour_lines.extend(neighbour_line)

            neighbour_lines_tensor = torch.tensor(neighbour_lines)
            input_tensor = torch.cat((current_line, neighbour_lines_tensor))
        else:
            input_tensor = current_line

        return {"input": input_tensor, "target": labels}


class JsonDatasetRenderer(JsonDataset):
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
        current_line_idx = self.data["lines"].index(self.data["lines"][self.ids[idx]])

        if self.neighbour_lines_cnt > 0:
            neighbor_indices = [
                current_line_idx + offset
                for delta in range(1, self.neighbour_lines_cnt + 1)
                for offset in (-delta, delta)
            ]
            neighbor_indices = sorted(neighbor_indices, key=lambda x: x)

            neighbour_lines = []
            for neighbor_idx in neighbor_indices:
                neighbour_line = self.get_line_features(neighbor_idx, current_line_idx)
                neighbour_lines.extend(neighbour_line)

            neighbour_lines_tensor = torch.tensor(neighbour_lines)
            input_tensor = torch.cat((current_line, neighbour_lines_tensor))
        else:
            input_tensor = current_line

        paddings = ["padding_top", "padding_right", "padding_bottom", "padding_left"]
        paddings = [padding + "_not_normalized" for padding in paddings]
        not_normalized_padding = torch.tensor([self.data["lines"][self.ids[idx]][key] for key in paddings])
        return {"input": input_tensor, "target": labels, "padding": not_normalized_padding, "page_id": self.data["lines"][self.ids[idx]]["page_id"]}



class TransformerEncoderNet(nn.Module):
    def __init__(self, classes, input_keys, positional_encoding, max_len):
        super().__init__()
        self.classes = classes
        self.input_keys = input_keys
        self.positionl_encoding = positional_encoding
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        upsampled_size = 128
        self.fc1 = nn.Linear(len(input_keys), upsampled_size)
        if self.positionl_encoding == "1d-page":
            self.pe = PositionalEncoding1d(upsampled_size, self.device, max_len)
        elif self.positionl_encoding == "1d-seq":
            self.pe = PositionalEncoding(upsampled_size, self.device, max_len=max_len)
        elif self.positionl_encoding == "2d":
            self.pe = PositionalEncoding2D(upsampled_size, self.device, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=upsampled_size, nhead=8)
        self.te = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc2 = nn.Linear(upsampled_size, len(classes))

    def forward(self, x):
        if self.positionl_encoding == "1d-page":
            position_x = x[:, :, -1]
            x = x[:, :, :-1]
        if self.positionl_encoding == "2d":
            positions_x = x[:, :, -2]
            positions_y = x[:, :, -1]
            x = x[:, :, :-2]
        x = F.leaky_relu(self.fc1(x))
        if self.positionl_encoding == "1d-page":
            pos_enc = self.pe(position_x).to(self.device)
            x += pos_enc
            x = x.permute(1, 0, 2)
        elif self.positionl_encoding == "1d-seq":
            x = x.permute(1, 0, 2)
            x = self.pe(x)
        elif self.positionl_encoding == "2d":
            pos_enc = self.pe(positions_x, positions_y).to(self.device)
            x += pos_enc
            x = x.permute(1, 0, 2)
        x = self.te(x)
        x = x.permute(1, 0, 2)
        x = torch.sigmoid(self.fc2(x))
        return x


class PositionalEncoding(nn.Module):
    # Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PositionalEncoding1d(nn.Module):
    # source https://github.com/tatp22/multidim-positional-encoding/tree/master
    # PE(x,2i) = sin(x/10000^(2i/D))
    # PE(x,2i+1) = cos(x/10000^(2i/D))

    # Where:
    # x is a point in 2d space
    # i is an integer in [0, D/2), where D is the size of the ch dimension
    def __init__(self, d_model: int, device, max_len: int = 1000):
        super().__init__()
        self.device = device
        if d_model % 2 != 0:
            d_model += 1
        
        pe = torch.zeros(max_len, d_model)
        for i in range(d_model // 2):
            for x in range(max_len):
                pe[x, 2*i] = math.sin(x / 10000**(2*i / d_model))
                pe[x, 2*i+1] = math.cos(x / 10000**(2*i / d_model))
        self.register_buffer('pe', pe)
        
    def forward(self, positions):
        batch_size, seq_len = positions.shape
        pos_enc = torch.zeros(batch_size, seq_len, self.pe.shape[1], dtype=self.pe.dtype, device=self.device)

        valid_positions = positions != -1
        valid_indices = positions[valid_positions].long()
        valid_encodings = self.pe[valid_indices]
        pos_enc[valid_positions] = valid_encodings

        return pos_enc

class PositionalEncoding2D(nn.Module):
    # source https://github.com/tatp22/multidim-positional-encoding/tree/master
    # PE(x,y,2i) = sin(x/10000^(4i/D))
    # PE(x,y,2i+1) = cos(x/10000^(4i/D))
    # PE(x,y,2j+D/2) = sin(y/10000^(4j/D))
    # PE(x,y,2j+1+D/2) = cos(y/10000^(4j/D))

    # Where:
    # (x,y) is a point in 2d space
    # i,j is an integer in [0, D/4), where D is the size of the ch dimension
    def __init__(self, d_model: int, device, max_len: int = 1000):
        super().__init__()
        if d_model % 4 != 0:
            d_model += 4 - d_model % 4
            
        self.device = device
            
        self.d_model = d_model
        pe = torch.zeros(max_len, max_len, d_model)
        for i in range(d_model // 4):
            for x in range(max_len):
                for y in range(max_len):
                    pe[x, y, 4*i] = math.sin(x / 10000**(4*i / d_model))
                    pe[x, y, 4*i+1] = math.cos(x / 10000**(4*i / d_model))
                    pe[x, y, 4*i+2] = math.sin(y / 10000**(4*i / d_model))
                    pe[x, y, 4*i+3] = math.cos(y / 10000**(4*i / d_model))
        self.register_buffer('pe', pe)
                    
    def forward(self, positions_x, positions_y):
        batch_size, seq_len = positions_x.shape
        pos_enc = torch.zeros(batch_size, seq_len, self.d_model, dtype=self.pe.dtype, device=self.device)

        valid_mask = (positions_x != -1) & (positions_y != -1)
        valid_indices_x = positions_x[valid_mask].long()
        valid_indices_y = positions_y[valid_mask].long()
        valid_encodings = self.pe[valid_indices_x, valid_indices_y]
        pos_enc[valid_mask] = valid_encodings
        
        return pos_enc


class JsonDatasetForTransformer(JsonDataset):
    def __init__(self, json_file, classes, input_keys, positional_encoding, neighbour_lines_cnt=0, max_len=1000, train=False, val=False):
        super().__init__(json_file, classes, input_keys, neighbour_lines_cnt, train, val)
        self.positional_encoding = positional_encoding
        self.max_len = max_len

    def get_line_features_and_labels(self, idx, current_line_idx):
        if 0 <= idx < len(self.data["lines"]) and \
           self.data["lines"][idx]["page_id"] == self.data["lines"][current_line_idx]["page_id"]:
            labels = self.data["lines"][idx]["labels"]
            x_position = self.data["lines"][idx][f"x_{self.max_len}"]
            y_position = self.data["lines"][idx][f"y_{self.max_len}"]
            return [self.data["lines"][idx][key] for key in self.input_keys], [labels.get(label, 0) for label in self.classes], x_position, y_position
        else:
            return [0] * len(self.input_keys), [0] * (len(self.classes) - 1) + [1], -1, -1

    def __getitem__(self, idx):
        current_line_features, labels, current_line_x, current_line_y = self.get_line_features_and_labels(self.ids[idx], self.ids[idx])
        input_tensor = torch.tensor(current_line_features).reshape(1, -1)
        labels = torch.tensor(labels, dtype=torch.float32)
        x_positions = torch.tensor([current_line_x])
        y_positions = torch.tensor([current_line_y])
        current_line_idx = self.data["lines"].index(self.data["lines"][self.ids[idx]])

        if self.neighbour_lines_cnt > 0:
            neighbor_indices = [
                current_line_idx + offset
                for delta in range(1, self.neighbour_lines_cnt + 1)
                for offset in (-delta, delta)
            ]
            neighbor_indices = sorted(neighbor_indices, key=lambda x: x)

            neighbour_lines = torch.tensor([])
            neighbour_labels = torch.tensor([], dtype=torch.int64)
            neighbour_x_positions = torch.tensor([])
            neighbour_y_positions = torch.tensor([])
            for i, neighbor_idx in enumerate(neighbor_indices):
                neighbour_line, neighbour_label, neighbour_x, neighbour_y = self.get_line_features_and_labels(neighbor_idx, current_line_idx)
                neighbour_lines = torch.cat((neighbour_lines, torch.tensor(neighbour_line).reshape(1, -1)))
                neighbour_labels = torch.cat((neighbour_labels, torch.tensor([neighbour_label], dtype=torch.float32)))
                neighbour_x_positions = torch.cat((neighbour_x_positions, torch.tensor([neighbour_x])))
                neighbour_y_positions = torch.cat((neighbour_y_positions, torch.tensor([neighbour_y])))

            input_tensor = torch.cat((neighbour_lines[:len(neighbour_lines)//2], input_tensor, neighbour_lines[len(neighbour_lines)//2:]))
            labels = labels.unsqueeze(0)
            labels = torch.cat((neighbour_labels[:len(neighbour_labels)//2, :], labels, neighbour_labels[len(neighbour_labels)//2:, :]))
            x_positions = torch.cat((neighbour_x_positions[:len(neighbour_x_positions)//2], x_positions, neighbour_x_positions[len(neighbour_x_positions)//2:]))
            y_positions = torch.cat((neighbour_y_positions[:len(neighbour_y_positions)//2], y_positions, neighbour_y_positions[len(neighbour_y_positions)//2:]))

        first_non_padding_position = -1
        last_non_padding_position = len(input_tensor)
        for i, label in enumerate(labels):
            if label[-1] == 0 and first_non_padding_position == -1:
                first_non_padding_position = i
            if label[-1] == 1 and first_non_padding_position != -1:
                last_non_padding_position = i
                break
        
        if self.positional_encoding == '1d-page':
            first_non_padding_position_on_page = 0
            for i in range(current_line_idx - len(labels) // 2, -1, -1):
                if self.data["lines"][i]["page_id"] != self.data["lines"][current_line_idx]["page_id"]:
                    break
                first_non_padding_position_on_page += 1
                
            positions = torch.full((input_tensor.shape[0], 1), -1)
            for i in range(first_non_padding_position, last_non_padding_position):
                positions[i] = first_non_padding_position_on_page + i
            input_tensor = torch.cat((input_tensor, positions), dim=1)
            
        if self.positional_encoding == '2d':
            input_tensor = torch.cat((input_tensor, x_positions.unsqueeze(1), y_positions.unsqueeze(1)), dim=1)

        return {"input": input_tensor, "target": labels}


def train_net(writer, net, trn_loader, val_loader, device, epochs, learning_rate, decay_start, decay_rate, decay_step):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    if decay_rate and decay_step:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)

    running_loss = 0.0
    for epoch in range(epochs):
        for i, data in enumerate(trn_loader, 0):
            inputs, targets = data["input"], data["target"]
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        writer.add_scalar("Train/loss", running_loss, epoch)
        running_loss = 0.0

        writer.add_scalar("Train/learning rate", optimizer.param_groups[0]["lr"], epoch)
        if decay_rate and decay_step and epoch >= decay_start - decay_step:
            scheduler.step()

        if epoch > 0 and epoch % (min(100, epochs // 10)) == 0:
            progress_path = os.path.join(os.path.dirname(args.model_path), f"progress_{epoch}.pt")
            torch.save(net.state_dict(), progress_path)

        if epoch % 10 == 9:
        # if True:
            thresholds = [0.5, 0.75, 0.9]
            val_thresholds_stats = test_positives_negatives(net, val_loader, device, thresholds)
            train_thresholds_stats = test_positives_negatives(net, trn_loader, device, thresholds)
            for threshold in thresholds:
                class_tp = val_thresholds_stats[threshold]["tp"]
                class_tn = val_thresholds_stats[threshold]["tn"]
                class_fp = val_thresholds_stats[threshold]["fp"]
                class_fn = val_thresholds_stats[threshold]["fn"]
                do_metrics(writer, epoch, "Val", class_tp, class_tn, class_fp, class_fn, threshold)
                class_tp = train_thresholds_stats[threshold]["tp"]
                class_tn = train_thresholds_stats[threshold]["tn"]
                class_fp = train_thresholds_stats[threshold]["fp"]
                class_fn = train_thresholds_stats[threshold]["fn"]                
                do_metrics(writer, epoch, "Train", class_tp, class_tn, class_fp, class_fn, threshold)


def do_metrics(writer, epoch, loader_type, class_tp, class_tn, class_fp, class_fn, threshold=0.75):
    precision, total_precision = test_precision(net, class_tp, class_fp)
    recall, total_recall = test_recall(net, class_tp, class_fn)
    f1, total_f1 = test_f1_score(net, precision, recall)
    accuracy, total_accuracy = test_accuracy(net, class_tp, class_tn, class_fp, class_fn)

    writer.add_scalar(f"{loader_type} threshold {threshold} precision", total_precision, epoch)
    writer.add_scalar(f"{loader_type} threshold {threshold} recall", total_recall, epoch)
    writer.add_scalar(f"{loader_type} threshold {threshold} f1", total_f1, epoch)
    writer.add_scalar(f"{loader_type} threshold {threshold} accuracy", total_accuracy, epoch)

    for i, class_name in enumerate(args.classes):
        writer.add_scalar(f"{loader_type} threshold {threshold} precision/{class_name}", precision[i], epoch)
        writer.add_scalar(f"{loader_type} threshold {threshold} recall/{class_name}", recall[i], epoch)
        writer.add_scalar(f"{loader_type} threshold {threshold} f1/{class_name}", f1[i], epoch)
        writer.add_scalar(f"{loader_type} threshold {threshold} accuracy/{class_name}", accuracy[i], epoch)


def test_positives_negatives(net, loader, device, thresholds=[0.75]):
    out = {threshold: {"tp": [0] * len(net.classes), "tn": [0] * len(net.classes), "fp": [0] * len(net.classes), "fn": [0] * len(net.classes)} for threshold in thresholds}
    with torch.no_grad():
        for data in loader:
            inputs, labels = data["input"], data["target"]
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            predicted_thresholds = {}
            for threshold in thresholds:
                predicted = (outputs >= threshold).int().view(-1, len(net.classes))
                predicted_thresholds[threshold] = predicted
            
            labels = labels.int().view(-1, len(net.classes))

            for threshold in thresholds:
                predicted = predicted_thresholds[threshold]
                class_tp = [0] * len(net.classes)
                class_tn = [0] * len(net.classes)
                class_fp = [0] * len(net.classes)
                class_fn = [0] * len(net.classes)
                for i in range(len(net.classes)):
                    class_tp[i] += ((predicted[:, i] == 1) & (labels[:, i] == 1)).sum().item()
                    class_tn[i] += ((predicted[:, i] == 0) & (labels[:, i] == 0)).sum().item()
                    class_fp[i] += ((predicted[:, i] == 1) & (labels[:, i] == 0)).sum().item()
                    class_fn[i] += ((predicted[:, i] == 0) & (labels[:, i] == 1)).sum().item()
                out[threshold]["tp"] = [out[threshold]["tp"][i] + class_tp[i] for i in range(len(net.classes))]
                out[threshold]["tn"] = [out[threshold]["tn"][i] + class_tn[i] for i in range(len(net.classes))]
                out[threshold]["fp"] = [out[threshold]["fp"][i] + class_fp[i] for i in range(len(net.classes))]
                out[threshold]["fn"] = [out[threshold]["fn"][i] + class_fn[i] for i in range(len(net.classes))]

    return out

def test_f1_score(net, precision, recall):
    f1 = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) if precision[i] + recall[i] > 0 else 0.0 for i in range(len(net.classes))]
    f1_without_padding = f1[:-1]
    total_f1 = sum(f1_without_padding) / len(f1_without_padding)
    return f1, total_f1


def test_recall(net, class_tp, class_fn):
    recall = [class_tp[i] / (class_tp[i] + class_fn[i]) if class_tp[i] + class_fn[i] != 0 else 0.0 for i in range(len(net.classes))]
    recall_without_padding = recall[:-1]
    total_recall = sum(recall_without_padding) / len(recall_without_padding)
    return recall, total_recall


def test_precision(net, class_tp, class_fp):
    precision = [class_tp[i] / (class_tp[i] + class_fp[i]) if class_tp[i] + class_fp[i] != 0 else 0.0 for i in range(len(net.classes))]
    precision_without_padding = precision[:-1]
    total_precision = sum(precision_without_padding) / len(precision_without_padding)
    return precision, total_precision


def test_accuracy(net, class_tp, class_tn, class_fp, class_fn):
    accuracy = [(class_tp[i] + class_tn[i]) / (class_tp[i] + class_tn[i] + class_fp[i] + class_fn[i]) if class_tp[i] + class_tn[i] + class_fp[i] + class_fn[i] != 0 else 0.0 for i in range(len(net.classes))]
    class_accuracy_without_padding = accuracy[:-1]
    total_accuracy = sum(class_accuracy_without_padding) / len(class_accuracy_without_padding)
    return accuracy, total_accuracy


def padding_to_bbox_pts(padding, img_width, img_height, bbox_padding=10):
    top_padding = padding[0]
    right_padding = padding[1]
    bottom_padding = padding[2]
    left_padding = padding[3]
    top_left = [left_padding, top_padding]
    top_right = [img_width - right_padding, top_padding]
    bottom_right = [img_width - right_padding, img_height - bottom_padding + bbox_padding]
    bottom_left = [left_padding, img_height - bottom_padding + bbox_padding]
    pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
    pts = pts.reshape((-1, 1, 2))
    return pts


def render_page_bboxes(loader, page_id, mastercopy_dir, output_render_dir, net, device, threshold=0.9):
    lines = []
    with torch.no_grad():
        for data in loader:
            if page_id not in data["page_id"]:
                continue
            inputs, labels = data["input"], data["target"]
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            predicted = (outputs >= threshold).int()

            labels = labels.int().view(-1, len(net.classes))
            predicted = predicted.view(-1, len(net.classes))

            lines.append({
                "paddings": data["padding"].tolist(),
                "predicted": predicted.tolist()
            })

    img_path = None
    for file in os.listdir(mastercopy_dir):
        if page_id in file:
            img_path = os.path.join(mastercopy_dir, file)
            break
    if img_path is None:
        print(f"Image for page {page_id} not found.")
        return
    img = cv2.imread(img_path)

    for line in lines:
        if len(line["predicted"]) == 0:
            continue
        predicted_classes = line["predicted"][0]
        for i, prediction in enumerate(predicted_classes):
            if prediction == 0:
                continue
            pts = padding_to_bbox_pts(line["paddings"], img.shape[1], img.shape[0])

            label = net.classes[i]
            if label == "kapitola":
                bbox_color = (255, 0, 0)
            elif label == "cislo strany":
                bbox_color = (0, 0, 255)
            else:
                bbox_color = (100, 100, 100)
            img = cv2.polylines(img, [pts], True, bbox_color, 2)
            img = cv2.putText(img, label, (pts[0][0][0], pts[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bbox_color, 2)

    page_id = page_id.split(".")[0]
    cv2.imwrite(os.path.join(output_render_dir, f'{page_id}_labeled.jpg'), img)


def render_val_pages(mastercopy_dir, output_render_dir, val_loader, net, device):
    os.makedirs(output_render_dir, exist_ok=True)
    for page_id in os.listdir(mastercopy_dir):
        page_id = page_id.split(".")[0]
        render_page_bboxes(val_loader, page_id, mastercopy_dir, output_render_dir, net, device)


def init_net_and_datasetloader_for_training(args):
    if args.net == "baseline":
        net = BaselineNet(args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt)
        val_data = JsonDataset(args.dataset, args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt, val=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        train_data = JsonDataset(args.dataset, args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt, train=True)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    elif args.net == "transformer":
        net = TransformerEncoderNet(args.classes, args.input_keys, args.positional_encoding, args.positional_encoding_max_len)
        val_data = JsonDatasetForTransformer(args.dataset, args.classes, args.input_keys, args.positional_encoding, max_len=args.positional_encoding_max_len, neighbour_lines_cnt=args.neighbour_lines_cnt, val=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        train_data = JsonDatasetForTransformer(args.dataset, args.classes, args.input_keys, args.positional_encoding, max_len=args.positional_encoding_max_len, neighbour_lines_cnt=args.neighbour_lines_cnt, train=True)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    return net, train_loader, val_loader

def comma_float(num):
    return f'{num:.3f}'.replace('.', ',')

def print_for_results(metric, num):
    print(f'{metric}: {comma_float(num)}')

if __name__ == "__main__":
    args = parse_args()
    args.classes += ["padding"]

    print("Classes:", args.classes)
    print("Input keys:", args.input_keys)
    print("Args:", args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net, trn_loader, val_loader = init_net_and_datasetloader_for_training(args)
    net.to(device)

    if args.epochs:
        writer_path = os.path.join(os.path.dirname(args.model_path), "summary_writer")
        writer = SummaryWriter(writer_path)

        train_net(writer, net, trn_loader, val_loader, device, args.epochs, args.learning_rate, args.decay_start, args.decay_rate, args.decay_step)
        torch.save(net.state_dict(), args.model_path)
    else:
        net.load_state_dict(torch.load(args.model_path, map_location=device))

    if args.render_val_images:
        val_loader = JsonDatasetRenderer(args.dataset, args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt)
        render_val_pages(args.mastercopy_dir, args.output_render_dir, val_loader, net, device)
        exit()

    thresholds = [0.5, 0.75, 0.9]
    val_thresholds_stats = test_positives_negatives(net, val_loader, device, thresholds)
    for threshold in thresholds:
        class_tp = val_thresholds_stats[threshold]["tp"]
        class_tn = val_thresholds_stats[threshold]["tn"]
        class_fp = val_thresholds_stats[threshold]["fp"]
        class_fn = val_thresholds_stats[threshold]["fn"]
        accuracy, total_accuracy = test_accuracy(net, class_tp, class_tn, class_fp, class_fn)
        precision, total_precision = test_precision(net, class_tp, class_fp)
        recall, total_recall = test_recall(net, class_tp, class_fn)
        f1, total_f1 = test_f1_score(net, precision, recall)
        
        print(80 * "=")        
        print("Threshold:", threshold)
        print_for_results("Total precision", total_precision)
        print_for_results("Total recall", total_recall)
        print_for_results("Total f1", total_f1)
        print_for_results("Total accuracy", total_accuracy)
        for i, class_name in enumerate(args.classes):
            print(20 * "*")
            print(f"{class_name}:")
            print_for_results("Precision", precision[i])
            print_for_results("Recall", recall[i])
            print_for_results("F1", f1[i])
            print_for_results("Accuracy", accuracy[i])
            
