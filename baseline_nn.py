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
        x = self.fc3(x)

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
        label = torch.tensor(map_label(self.classes, self.data["lines"][self.ids[idx]]["label"]))
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

        return {"input": input_tensor, "target": label}


class TransformerEncoderNet(nn.Module):
    def __init__(self, classes, input_keys, neighbour_lines_cnt=0):
        super().__init__()
        self.classes = classes
        self.input_keys = input_keys

        upsampled_size = 128
        self.fc1 = nn.Linear(len(input_keys), upsampled_size)
        self.pos_encoder = PositionalEncoding(upsampled_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=upsampled_size, nhead=8, batch_first=True)
        self.te = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc2 = nn.Linear(upsampled_size, len(classes))

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.pos_encoder(x)
        x = self.te(x)
        x = F.leaky_relu(self.fc2(x))
        x = x.view(-1, len(self.classes))
        return x

# source https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # Added unsqueeze to add a batch dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class JsonDatasetForTransformer(JsonDataset):
    def get_line_features_and_labels(self, idx, current_line_idx):
        if 0 <= idx < len(self.data["lines"]) and \
           self.data["lines"][idx]["page_id"] == self.data["lines"][current_line_idx]["page_id"]:
            return [self.data["lines"][idx][key] for key in self.input_keys], map_label(self.classes, self.data["lines"][idx]["label"])
        else:
            return [0] * len(self.input_keys), map_label(self.classes, "padding")
        

    def __getitem__(self, idx):
        labels = torch.tensor([map_label(self.classes, self.data["lines"][self.ids[idx]]["label"])])
        input_tensor = torch.tensor(self.get_line_features(self.ids[idx], self.ids[idx])).reshape(1, -1)
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
            for neighbor_idx in neighbor_indices:
                neighbour_line, neighbour_label = self.get_line_features_and_labels(neighbor_idx, current_line_idx)
                neighbour_lines = torch.cat((neighbour_lines, torch.tensor(neighbour_line).reshape(1, -1)))
                neighbour_labels = torch.cat((neighbour_labels, torch.tensor([neighbour_label])))
                
            input_tensor = torch.cat((neighbour_lines[:len(neighbour_lines)//2], input_tensor, neighbour_lines[len(neighbour_lines)//2:]))
            labels = torch.cat((neighbour_labels[:len(neighbour_labels)//2], labels, neighbour_labels[len(neighbour_labels)//2:]))
            
        return {"input": input_tensor, "target": labels}


def test_accuracy(net, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data["input"], data["target"]
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1)

            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)           

            # Ignore padding
            mask = labels != net.classes.index("padding")
            total += mask.sum().item()
            correct += (predicted[mask] == labels[mask]).sum().item()

    if total == 0:
        return 0.0
    return correct / total * 100


def test_class_accuracy(net, loader, device):
    class_correct = [0] * len(net.classes)
    class_total = [0] * len(net.classes)

    with torch.no_grad():
        for data in loader:
            inputs, labels = data["input"], data["target"]
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1)

            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)

            c = (predicted == labels)
            if c.dim() > 1:
                c = c.squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    return [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 for i in range(len(net.classes))]


def train_net(writer, net, trn_loader, val_loader, device, epochs, learning_rate, decay_start, decay_rate, decay_step):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    if decay_rate and decay_step:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)

    running_loss = 0.0
    for epoch in range(epochs):
        for i, data in enumerate(trn_loader, 0):
            inputs, targets = data["input"], data["target"]
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(-1)

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

        if epochs // 10 > 0 and epoch % (epochs // 10) == 0:
            progress_path = os.path.join(os.path.dirname(args.model_path), f"progress_{epoch}.pt")
            torch.save(net.state_dict(), progress_path)

        if epoch % 10 == 9:
            writer.add_scalar("Train accuracy", test_accuracy(net, trn_loader, device), epoch)
            for i, class_accuracy in enumerate(test_class_accuracy(net, trn_loader, device)):
                writer.add_scalar(f"Train accuracy/{net.classes[i]}", class_accuracy, epoch)
            writer.add_scalar("Val accuracy", test_accuracy(net, val_loader, device), epoch)
            for i, class_accuracy in enumerate(test_class_accuracy(net, val_loader, device)):
                writer.add_scalar(f"Val accuracy/{net.classes[i]}", class_accuracy, epoch)


def padding_to_bbox_pts(padding, img_width, img_height, bbox_padding=10):
    top_padding = padding[0]
    bottom_padding = padding[1]
    left_padding = padding[2]
    right_padding = padding[3]
    top_left = [left_padding, top_padding]
    top_right = [img_width - right_padding, top_padding]
    bottom_right = [img_width - right_padding, img_height - bottom_padding + bbox_padding]
    bottom_left = [left_padding, img_height - bottom_padding + bbox_padding]
    pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
    pts = pts.reshape((-1, 1, 2))
    return pts


def render_page_bboxes(loader, page_id, mastercopy_dir, output_render_dir, net, device):
    lines = []
    with torch.no_grad():
        for data in loader:
            inputs, labels = data["input"], data["target"]
            inputs, labels = inputs.to(device), labels.to(device)

            output = net(inputs)
            _, data["predicted"] = torch.max(output, 1)

            lines.append(data)

    img = cv2.imread(os.path.join(mastercopy_dir, page_id + ".jpg"))

    for line in lines:
        pts = padding_to_bbox_pts(line["input"][-4:], img.shape[1], img.shape[0])

        label = net.classes[line["predicted"]]
        if label == "kapitola":
            bbox_color = (255, 0, 0)
        elif label == "cislo strany":
            bbox_color = (0, 0, 255)
        else:
            bbox_color = (0, 255, 0)
        img = cv2.polylines(img, [pts], True, bbox_color, 2)

    page_id = page_id.split(".")[0]
    cv2.imwrite(os.path.join(output_render_dir, f'{page_id}_labeled.jpg'), img)


def render_val_pages(mastercopy_dir, output_render_dir, dataset, input_keys, net, device):
    os.makedirs(output_render_dir, exist_ok=True)
    for page_id in os.listdir(mastercopy_dir):
        if not page_id.endswith(".jpg"):
            continue
        val_data = JsonDataset(dataset, args.classes, input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt, val=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)
        render_page_bboxes(val_loader, page_id, mastercopy_dir, output_render_dir, net, device)


def init_net_and_datasetloader_for_training(net_type):
    if net_type == "baseline":
        net = BaselineNet(args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt)
        val_data = JsonDataset(args.dataset, args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt, val=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        train_data = JsonDataset(args.dataset, args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt, train=True)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    elif net_type == "transformer":
        net = TransformerEncoderNet(args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt)
        val_data = JsonDatasetForTransformer(args.dataset, args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt, val=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        train_data = JsonDatasetForTransformer(args.dataset, args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt, train=True)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    return net, train_loader, val_loader


if __name__ == "__main__":
    args = parse_args()
    args.classes += ["padding"]

    print("Classes:", args.classes)
    print("Input keys:", args.input_keys)
    print("Args:", args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.epochs:
        net, trn_loader, val_loader = init_net_and_datasetloader_for_training(args.net)
        net.to(device)

        writer_path = os.path.join(os.path.dirname(args.model_path), "summary_writer")
        writer = SummaryWriter(writer_path)

        train_net(writer, net, trn_loader, val_loader, device, args.epochs, args.learning_rate, args.decay_start, args.decay_rate, args.decay_step)
        torch.save(net.state_dict(), args.model_path)
    else:
        if args.net == "baseline":
            net = BaselineNet(args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt)
        elif args.net == "transformer":
            net = TransformerEncoderNet(args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt)
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        
    if args.render_val_images:
        render_val_pages(args.mastercopy_dir, args.output_render_dir, args.dataset, args.input_keys, net, device)

    print(f"Val accuracy: {test_accuracy(net, val_loader, device)}")
    for i, class_accuracy in enumerate(test_class_accuracy(net, val_loader, device)):
        print(f"Val accuracy for {args.classes[i]}: {class_accuracy}")
