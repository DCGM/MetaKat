import argparse
import os
import json
import numpy as np
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
    
    parser.add_argument("--render-page-count", type=int)
    parser.add_argument("--mastercopy-dir")
    parser.add_argument("--output-render-dir")
    
    args = parser.parse_args()
    
    args.classes = args.classes.split(";")
    return args

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

class TransformerEncoderNet(nn.Module):
    def __init__(self, classes, input_keys, neighbour_lines_cnt=0):
        super().__init__()
        self.classes = classes
        self.input_keys = input_keys
        self.input_size = len(input_keys) * (neighbour_lines_cnt * 2 + 1)
        
        upsampled_size = 128
        self.fc1 = nn.Linear(self.input_size, upsampled_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=upsampled_size, nhead=8)
        self.te = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc2 = nn.Linear(upsampled_size, len(classes))

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.te(x)
        x = self.fc2(x)

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
        
        return {"input": input_tensor, "label": label}

# TODO remove
class JsonDatasetFull(JsonDataset):
    def __getitem__(self, idx):
        line = self.data["lines"][self.ids[idx]]
        label = torch.tensor(map_label(self.classes, line["label"]))
        input = torch.tensor([line[key] for key in self.input_keys])
        
        # TODO fix
        return {"label":label, "input":input, "id":self.ids[idx], "page_id":self.data["lines"][self.ids[idx]]["page_id"]}

    def filter_by_page_id(self, page_id):
        self.data["lines"] = [line for line in self.data["lines"] if line["page_id"] == page_id]

def map_label(classes, label):
    return classes.index(label)

def test_accuracy(net, loader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            inputs, labels = data["input"], data["label"]
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if total == 0:
        return 0.0
    return correct / total * 100

def test_class_accuracy(net, loader, device):
    class_correct = [0] * len(net.classes)
    class_total = [0] * len(net.classes)
    
    with torch.no_grad():
        for data in loader:
            inputs, labels = data["input"], data["label"]
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            
            c = (predicted == labels)
            if c.dim() > 1:
                c = c.squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    return  [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 for i in range(len(net.classes))]
        

def train_net(writer, net, trn_loader, val_loader, device, epochs, learning_rate, decay_start, decay_rate, decay_step):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    if decay_rate and decay_step:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)

    running_loss = 0.0
    for epoch in range(epochs):        
        for i, data in enumerate(trn_loader, 0):
            inputs, labels = data["input"], data["label"]
            inputs, labels = inputs.to(device), labels.to(device)
               
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        writer.add_scalar("Train/loss", running_loss, epoch)
        running_loss = 0.0
        
        writer.add_scalar("Train/learning rate", optimizer.param_groups[0]["lr"], epoch)
        if decay_rate and decay_step and epoch >= decay_start - decay_step:
            scheduler.step()
        
        if epoch % 1000 == 999:
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
            inputs, labels = data["input"], data["label"]
            inputs, labels = inputs.to(device), labels.to(device)
            
            output = net(inputs)
            _, data["predicted"] = torch.max(output, 1)
            
            lines.append(data)
        
    img = cv2.imread(os.path.join(mastercopy_dir, page_id + ".jpg"))
  
    for line in lines:
        pts = padding_to_bbox_pts(line["input"][0][0][4:8], img.shape[1], img.shape[0])
        
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

def render_dir_bboxes(mastercopy_dir, output_render_dir, render_page_count, dataset, input_keys, net, device):
    os.makedirs(output_render_dir, exist_ok=True)
    for page_id in os.listdir(mastercopy_dir)[:render_page_count]:
        page_id = ".".join(page_id.split(".")[:-1])
        full_dataset = JsonDatasetFull(dataset, args.classes, input_keys)
        full_dataset.filter_by_page_id(page_id)
        full_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)
        render_page_bboxes(full_loader, page_id, mastercopy_dir, output_render_dir, net, device)

if __name__ == "__main__":
    args = parse_args()
    
    print("Classes:", args.classes)
    print("Input keys:", args.input_keys)
    print("Args:", args)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.net == "baseline":
        net = BaselineNet(args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt)
    elif args.net == "transformer":
        net = TransformerEncoderNet(args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt)
    
    net.to(device)
    
    if args.render_page_count and args.mastercopy_dir and args.output_render_dir:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        render_dir_bboxes(args.mastercopy_dir, args.output_render_dir, args.render_page_count, args.dataset, args.input_keys, net, device)
        exit()

    val_data = JsonDataset(args.dataset, args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt, val=True)    
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    if args.epochs:
        trn_data = JsonDataset(args.dataset, args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt, train=True)
        trn_loader = DataLoader(trn_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)   
        
        writer_path = os.path.join(os.path.dirname(args.model_path), "summary_writer")
        writer = SummaryWriter(writer_path)
             
        train_net(writer, net, trn_loader, val_loader, device, args.epochs, args.learning_rate, args.decay_start, args.decay_rate, args.decay_step)
        torch.save(net.state_dict(), args.model_path)
    else:
        net.load_state_dict(torch.load(args.model_path, map_location=device))    
    
    print(f"Val accuracy: {test_accuracy(net, val_loader, device)}")
    for i, class_accuracy in enumerate(test_class_accuracy(net, val_loader, device)):
        print(f"Val accuracy for {args.classes[i]}: {class_accuracy}")
            
