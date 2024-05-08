# File: train_utils.py
# Author: Jakub Křivánek
# Date: 7. 5. 2024
# Description: This file contains the training utilities for the FCNN and TENN models.

import argparse
import os
import json
import numpy as np
import math
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from digilinka.nets.fcnn import FCNN, JsonDataset, JsonDatasetRenderer
from digilinka.nets.tenn import TENN, JsonDatasetForTENN

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--net", choices=["fcnn", "tenn"], required=True)
    parser.add_argument("--dataset", required=True)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--decay-start", type=int, default=200)
    parser.add_argument("--decay-rate", type=float, default=0.0)
    parser.add_argument("--decay-step", type=int, default=0)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--input-keys", nargs="+", required=True)
    parser.add_argument("--neighbour-lines-cnt", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model-path", required=True)
    parser.add_argument('--positional-encoding', choices=['2d', '1d-page', '1d-seq', '1d-seq-2d'], default='2d', help='Possitional encoding for the model')
    parser.add_argument('--positional-encoding-max-len', type=int, default=1000, help='Max len for positional encoding')

    parser.add_argument("--render-val-images", action="store_true")
    parser.add_argument("--mastercopy-dir")
    parser.add_argument("--output-render-dir")

    args = parser.parse_args()

    args.classes = args.classes.split(";")
    return args


def map_label(classes, label):
    return classes.index(label)


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
                do_metrics("Val", class_tp, class_tn, class_fp, class_fn, threshold=threshold, writer=writer, epoch=epoch)
                class_tp = train_thresholds_stats[threshold]["tp"]
                class_tn = train_thresholds_stats[threshold]["tn"]
                class_fp = train_thresholds_stats[threshold]["fp"]
                class_fn = train_thresholds_stats[threshold]["fn"]                
                do_metrics("Train", class_tp, class_tn, class_fp, class_fn, threshold=threshold, writer=writer, epoch=epoch)


def do_metrics(loader_type, class_tp, class_tn, class_fp, class_fn, threshold=0.75, writer=None, epoch=None):
    precision, total_precision = test_precision(net, class_tp, class_fp)
    recall, total_recall = test_recall(net, class_tp, class_fn)
    f1, total_f1 = test_f1_score(net, precision, recall)
    accuracy, total_accuracy = test_accuracy(net, class_tp, class_tn, class_fp, class_fn)
    
    if writer is not None and epoch is not None:
        writer.add_scalar(f"{loader_type} threshold {threshold} precision", total_precision, epoch)
        writer.add_scalar(f"{loader_type} threshold {threshold} recall", total_recall, epoch)
        writer.add_scalar(f"{loader_type} threshold {threshold} f1", total_f1, epoch)
        writer.add_scalar(f"{loader_type} threshold {threshold} accuracy", total_accuracy, epoch)

        for i, class_name in enumerate(net.classes):
            writer.add_scalar(f"{loader_type} threshold {threshold} precision/{class_name}", precision[i], epoch)
            writer.add_scalar(f"{loader_type} threshold {threshold} recall/{class_name}", recall[i], epoch)
            writer.add_scalar(f"{loader_type} threshold {threshold} f1/{class_name}", f1[i], epoch)
            writer.add_scalar(f"{loader_type} threshold {threshold} accuracy/{class_name}", accuracy[i], epoch)
    else:
        print(f"Metrices for {loader_type} threshold {threshold}")
        print_for_results("precision", total_precision)
        print_for_results("recall", total_recall)
        print_for_results("f1", total_f1)
        print_for_results("accuracy", total_accuracy)
        for i, class_name in enumerate(net.classes):
            print(40 * "*")
            print(f"Class {class_name}")
            print_for_results("precision", precision[i])
            print_for_results("recall", recall[i])
            print_for_results("f1", f1[i])
            print_for_results("accuracy", accuracy[i])


def test_positives_negatives(net, loader, device, thresholds=[0.75]):
    # count_one -- count only the middle line = the line "chosen by loader"
    count_one = True if isinstance(net, TENN) else False
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
                    if count_one:
                        class_tp[i] += ((predicted[len(predicted) // 2:, i] == 1) & (labels[len(labels) // 2:, i] == 1)).sum().item()
                        class_tn[i] += ((predicted[len(predicted) // 2:, i] == 0) & (labels[len(labels) // 2:, i] == 0)).sum().item()
                        class_fp[i] += ((predicted[len(predicted) // 2:, i] == 1) & (labels[len(labels) // 2:, i] == 0)).sum().item()
                        class_fn[i] += ((predicted[len(predicted) // 2:, i] == 0) & (labels[len(labels) // 2:, i] == 1)).sum().item()                    
                    else:
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


def init_net_and_datasetloader_for_training(args, device):
    if args.net == "fcnn":
        net = FCNN(args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt)
        val_data = JsonDataset(args.dataset, args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt, val=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        train_data = JsonDataset(args.dataset, args.classes, args.input_keys, neighbour_lines_cnt=args.neighbour_lines_cnt, train=True)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    elif args.net == "tenn":
        net = TENN(args.classes, args.input_keys, args.positional_encoding, args.positional_encoding_max_len, device)
        val_data = JsonDatasetForTENN(args.dataset, args.classes, args.input_keys, args.positional_encoding, max_len=args.positional_encoding_max_len, neighbour_lines_cnt=args.neighbour_lines_cnt, val=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        train_data = JsonDatasetForTENN(args.dataset, args.classes, args.input_keys, args.positional_encoding, max_len=args.positional_encoding_max_len, neighbour_lines_cnt=args.neighbour_lines_cnt, train=True)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
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

    net, trn_loader, val_loader = init_net_and_datasetloader_for_training(args, device)
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
        print(80*"=")
        do_metrics("Val", class_tp, class_tn, class_fp, class_fn, threshold=threshold)
            
