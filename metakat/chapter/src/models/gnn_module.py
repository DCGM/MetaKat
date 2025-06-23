# File containing GNN model class.
# Author: Richard Blažo
# File name: gnn_module.py
# Description: This file contains the GNN model class and its methods.
# Parts of the code were designed with the help of ChatGPT.
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# This class was designed with use of ChatGPT


class MultiTaskGNN(torch.nn.Module):
    def __init__(self, input_dim=9, hidden_dim=32, node_classes=2):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.node_classifier = torch.nn.Linear(hidden_dim, node_classes)
        self.edge_scorer = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, edge_label_index=None):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        node_logits = self.node_classifier(x)

        edge_logits = None
        if edge_label_index is not None:
            src = x[edge_label_index[0]]
            dst = x[edge_label_index[1]]
            edge_inputs = torch.cat([src, dst], dim=1)
            edge_logits = self.edge_scorer(edge_inputs)

        return node_logits, edge_logits
