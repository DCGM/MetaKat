# GNN Training Script
# Author: Richard Blažo
# File name: gnn_train.py
# Description: Script used to train a GNN model for chapter and subchapter classification.
# Parts of code were designed with the use of ChatGPT.

import argparse
import json
import os
from collections import Counter

import torch
from sklearn.metrics import (classification_report,
                             precision_recall_fscore_support)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from src.utils.build_training_data import build_training_graph_gnn
from src.models.gnn_module import MultiTaskGNN

parser = argparse.ArgumentParser(description="GNN Training Script")
parser.add_argument("--dataset", type=str, default="./data/GNN/digilinka.json",
                    help="Path to the dataset file")
parser.add_argument("--batch_size", type=int, default=1,
                    help="Batch size for training")
parser.add_argument("--epochs", type=int, default=125,
                    help="Number of epochs for training")
parser.add_argument("--eval_epochs", type=int, default=25,
                    help="Number of epochs between evaluations")
parser.add_argument("--save_model_path", type=str, default="./models/GNN/",
                    help="Path to save the trained model")
parser.add_argument("--save_between", action="store_true",
                    help="Save the model between epochs")
parser.add_argument("--model_name", type=str, default="chapter_classifier_gnn.pth",
                    help="Name of the model to save")
args = parser.parse_args()


def build_datasets(json_path):
    with open(json_path, "r") as f:
        pages = json.load(f)

    graphs = build_training_graph_gnn(pages)
    dataset = GraphDataset(graphs)

    train, val = random_split(
        dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    return train, val


class GraphDataset(Dataset):
    def __init__(self, graphs):
        super().__init__()
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            node_logits, _ = model(batch.x, batch.edge_index)

            pred_labels = node_logits.argmax(dim=1).cpu().tolist()
            true_labels = batch.y.cpu().tolist()

            all_preds.extend(pred_labels)
            all_labels.extend(true_labels)

    print(classification_report(all_labels, all_preds,
          target_names=["chapter", "subchapter"], zero_division=0))

    results = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", labels=["chapter", "subchapter"], zero_division=0)

    return results


def predict_edges(embeddings, candidate_edges):
    src = embeddings[candidate_edges[0]]
    dst = embeddings[candidate_edges[1]]
    scores = model.edge_scorer(src, dst).squeeze()

    if scores.dim() == 0:
        scores = scores.unsqueeze(0)
    return torch.sigmoid(scores)


# Following part of the code was designed with the use of ChatGPT
if __name__ == "__main__":
    train_dataset, val_dataset = build_datasets(args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4)
    print(f"Loaded {len(train_dataset)} graphs.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Train label distribution:", Counter(
        [label for graph in train_dataset for label in graph.y.tolist()]))
    print("Val label distribution:", Counter(
        [label for graph in val_dataset for label in graph.y.tolist()]))

    model = MultiTaskGNN()
    optimizer = Adam(model.parameters(), lr=0.01)
    decayer = StepLR(optimizer, step_size=50, gamma=0.5)
    criterion_node = CrossEntropyLoss(
        weight=torch.tensor([1.0, 5.0]).to(device))
    criterion_edge = BCEWithLogitsLoss()

    model = model.to(device)

    evalEpochs = args.eval_epochs
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

    best_version = 0

    model.train()
    for epoch in range(args.epochs):
        total_node_loss = 0
        total_edge_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            node_logits, edge_logits = model(
                batch.x, batch.edge_index, batch.edge_label_index)

            node_labels = batch.y

            loss_node = criterion_node(node_logits, node_labels)
            loss_edge = criterion_edge(
                edge_logits.view(-1), batch.edge_labels.float())
            loss = loss_node + loss_edge
            loss.backward()
            optimizer.step()

            total_node_loss = loss_node.item()
            total_edge_loss += loss_edge.item()

        decayer.step()

        print(
            f"Epoch {epoch+1} - Node Loss: {total_node_loss:.4f}")
        if epoch % evalEpochs == 0:
            results = evaluate(model, val_loader, device)
            node_precision, node_recall, node_f1, _ = results
            if node_f1 > best_f1:
                best_version = epoch
                best_f1 = node_f1
                best_precision = node_precision
                best_recall = node_recall
                print(
                    f"New best F1: {best_f1:.4f} (Precision: {best_precision:.4f}, Recall: {best_recall:.4f})")
                if args.save_between:
                    torch.save(model.state_dict(), os.path.join(
                        args.save_model_path, f"model_temp_{epoch}.pth"))
                    print(f"Model saved as model_temp_{epoch}.pth")

    evaluate(model, val_loader, device)
    torch.save(model.state_dict(), os.path.join(
        args.save_model_path, args.model_name))
    print(f"Model saved to {args.save_model_path}{args.model_name}")
