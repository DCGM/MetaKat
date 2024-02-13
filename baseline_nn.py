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

classes = ["kapitola", "cislo strany", "text"]
input_keys = ["line_width", "line_height", "relative_line_width", "relative_line_height", "padding_top", "padding_bottom", "padding_left", "padding_right", "transcription_length"]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.00001)
    parser.add_argument("--decay-rate", type=float, default=0.0)
    parser.add_argument("--decay-step", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model-path", required=True)
    
    parser.add_argument("--render-page-count", type=int)
    parser.add_argument("--mastercopy-dir")
    parser.add_argument("--output-render-dir")
        
    return parser.parse_args()

class BaselineNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 6, 3)
        self.conv2 = nn.Conv1d(6, 16, 3)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(16 * 2, 120)
        self.fc2 = nn.Linear(120, 3)
               
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
        
def map_label(label):
    return classes.index(label)
    
class JsonDataset(Dataset):
    def __init__(self, json_file, train=False, test=False):
        with open(json_file) as f:
            self.data = json.load(f)
        
        if train:
            self.data = self.data["train"]
        elif test:
            self.data = self.data["test"]
        else:
            try:
                self.data = self.data["train"].extend(self.data["test"])
            except KeyError:
                self.data = self.data
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label = torch.tensor(map_label(self.data[idx]["label"]))
        input = torch.tensor([self.data[idx][key] for key in input_keys])
        input = input.unsqueeze(0)
        
        return {"label":label, "input":input}
    
class JsonDatasetFull(JsonDataset):
    def __getitem__(self, idx):
        label = torch.tensor(map_label(self.data[idx]["label"]))
        input = torch.tensor([self.data[idx][key] for key in input_keys])
        input = input.unsqueeze(0)
        
        return {"label":label, "input":input, "id":self.data[idx]["page_id"]}

    def filter_by_page_id(self, page_id):
        self.data = [line for line in self.data if line["page_id"] == page_id]

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
        return 0    
    return correct / total * 100

def test_class_accuracy(net, loader, device):
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    
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
    
    return [100 * class_correct[i] / class_total[i] for i in range(len(classes)) if class_total[i] > 0]
        

def train_net(writer, net, trn_loader, tst_loader, device, epochs, learning_rate, decay_rate, decay_step):
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
            if i % 20 == 19:
                writer.add_scalar("Loss/train", running_loss / 20, epoch * len(trn_loader) + i)
                running_loss = 0.0
        
        if decay_rate and decay_step and epoch > 1000:
            scheduler.step()
        
        if epoch % 1000 == 999:
            progress_path = os.path.join(os.path.dirname(args.model_path), f"progress_{epoch}.pt")
            torch.save(net.state_dict(), progress_path)
		
        if epoch % 10 == 9:
            writer.add_scalar("Accuracy/train", test_accuracy(net, trn_loader, device), epoch)
            writer.add_scalar("Accuracy/test", test_accuracy(net, tst_loader, device), epoch)
            for i, class_accuracy in enumerate(test_class_accuracy(net, trn_loader, device)):
                writer.add_scalar(f"Accuracy/train/{classes[i]}", class_accuracy, epoch)
            for i, class_accuracy in enumerate(test_class_accuracy(net, tst_loader, device)):
                writer.add_scalar(f"Accuracy/test/{classes[i]}", class_accuracy, epoch)
           

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
        
        label = classes[line["predicted"]]
        if label == "kapitola":
            bbox_color = (255, 0, 0)
        elif label == "cislo strany":
            bbox_color = (0, 0, 255)
        else:
            bbox_color = (0, 255, 0)
        img = cv2.polylines(img, [pts], True, bbox_color, 2)        
    
    page_id = page_id.split(".")[0]
    cv2.imwrite(os.path.join(output_render_dir, f'{page_id}_labeled.jpg'), img)

def render_dir_bboxes(mastercopy_dir, output_render_dir, render_page_count, dataset, net, device):
    os.makedirs(output_render_dir, exist_ok=True)
    for page_id in os.listdir(mastercopy_dir)[:render_page_count]:
        page_id = ".".join(page_id.split(".")[:-1])
        full_dataset = JsonDatasetFull(dataset)
        full_dataset.filter_by_page_id(page_id)
        full_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)
        render_page_bboxes(full_loader, page_id, mastercopy_dir, output_render_dir, net, device)

if __name__ == "__main__":
    args = parse_args()
    
    print("Classes:", classes)
    print("Input keys:", input_keys)
    print("Args:", args)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = BaselineNet()
    
    net.to(device)
    
    if args.render_page_count and args.mastercopy_dir and args.output_render_dir:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        render_dir_bboxes(args.mastercopy_dir, args.output_render_dir, args.render_page_count, args.dataset, net, device)
        exit()

    tst_data = JsonDataset(args.dataset, test=True)    
    tst_loader = DataLoader(tst_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    if args.epochs:
        trn_data = JsonDataset(args.dataset, train=True)
        trn_loader = DataLoader(trn_data, batch_size=args.batch_size, shuffle=True, num_workers=8)   
        
        writer_path = os.path.join(os.path.dirname(args.model_path), "summary_writer")
        writer = SummaryWriter(writer_path)
             
        train_net(writer, net, trn_loader, tst_loader, device, args.epochs, args.learning_rate, args.decay_rate, args.decay_step)
        torch.save(net.state_dict(), args.model_path)
    else:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
    
    
    print(f"Test accuracy: {test_accuracy(net, tst_loader, device)}")
    for i, class_accuracy in enumerate(test_class_accuracy(net, tst_loader, device)):
        print(f"Test accuracy for {classes[i]}: {class_accuracy}")
            
