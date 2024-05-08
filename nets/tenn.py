# File: tenn.py
# Author: Jakub Křivánek
# Date: 7. 5. 2024
# Description: This file contains the TENN model, positional encodings and the dataset class for the TENN model.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fcnn import JsonDataset

class TENN(nn.Module):
    def __init__(self, classes, input_keys, positional_encoding, max_len, device):
        super().__init__()
        self.classes = classes
        self.input_keys = input_keys
        self.positionl_encoding = positional_encoding
        self.device = device

        upsampled_size = 128
        self.fc1 = nn.Linear(len(input_keys), upsampled_size)
        if self.positionl_encoding == "1d-seq":
            self.pe = PositionalEncoding(upsampled_size, max_len=max_len)
        elif self.positionl_encoding == "1d-page":
            self.pe = PositionalEncoding1d(upsampled_size, self.device, max_len=max_len)        
        elif self.positionl_encoding == "2d":
            self.pe = PositionalEncoding2D(upsampled_size, self.device, max_len=max_len)
        elif self.positionl_encoding == "1d-seq-2d":
            self.pe_1d = PositionalEncoding(upsampled_size, max_len=max_len)
            self.pe_2d = PositionalEncoding2D(upsampled_size, self.device, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=upsampled_size, nhead=8, batch_first=True)
        self.te = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc2 = nn.Linear(upsampled_size, len(classes))

    def forward(self, x):
        if self.positionl_encoding == "1d-page":
            positions_x = x[:, :, -1]
            x = x[:, :, :-1]
        if self.positionl_encoding == "2d" or self.positionl_encoding == "1d-seq-2d":
            positions_x = x[:, :, -2]
            positions_y = x[:, :, -1]
            x = x[:, :, :-2]
            
        x = F.leaky_relu(self.fc1(x))
        
        if self.positionl_encoding == "1d-seq":
            x = x.permute(1, 0, 2)
            x = self.pe(x)
            x = x.permute(1, 0, 2)
        else:
            if self.positionl_encoding == "1d-page":
                pos_enc = self.pe(positions_x).to(self.device) 
            elif self.positionl_encoding == "2d":
                pos_enc = self.pe(positions_x, positions_y).to(self.device)
            elif self.positionl_encoding == "1d-seq-2d":
                pos_enc = self.pe_2d(positions_x, positions_y).to(self.device)
                x = x.permute(1, 0, 2)
                x = self.pe_1d(x)
                x = x.permute(1, 0, 2)
            x += pos_enc
        x = self.te(x)
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
            raise ValueError("d_model must be divisible by 2")
        
        self.d_model = d_model
        self.device = device
        positions = torch.arange(0, max_len, dtype=torch.float32, device=device)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2, device=device) / d_model)

        pe = torch.zeros(max_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(positions.unsqueeze(-1) / div_term)
        pe[:, 1::2] = torch.cos(positions.unsqueeze(-1) / div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, positions):
        batch_size, seq_len = positions.shape
        pos_enc = torch.zeros(batch_size, seq_len, self.d_model, dtype=self.pe.dtype, device=self.device)
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
            raise ValueError("d_model must be divisible by 4")
        
        self.d_model = d_model
        self.device = device
        pe = torch.zeros(max_len, max_len, d_model, device=device)
        positions = torch.arange(0, max_len, dtype=torch.float32, device=device)
        div_term = torch.pow(10000, torch.arange(0, d_model / 4, device=device) * 4 / d_model)

        pe[:, :, 0::4] = torch.sin(positions.unsqueeze(-1) / div_term).unsqueeze(1)
        pe[:, :, 1::4] = torch.cos(positions.unsqueeze(-1) / div_term).unsqueeze(1)
        pe[:, :, 2::4] = torch.sin(positions.unsqueeze(-1) / div_term).unsqueeze(0)
        pe[:, :, 3::4] = torch.cos(positions.unsqueeze(-1) / div_term).unsqueeze(0)

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


class JsonDatasetForTENN(JsonDataset):
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
        if self.positional_encoding in ['2d', '1d-seq-2d']:
            x_positions = torch.tensor([current_line_x], dtype=torch.int64)
            y_positions = torch.tensor([current_line_y], dtype=torch.int64)

        if self.neighbour_lines_cnt > 0:
            neighbor_indices = [
                idx + offset
                for delta in range(1, self.neighbour_lines_cnt + 1)
                for offset in (-delta, delta)
            ]
            neighbor_indices = sorted(neighbor_indices, key=lambda x: x)

            neighbour_lines = torch.tensor([])
            neighbour_labels = torch.tensor([], dtype=torch.int64)
            if self.positional_encoding in ['2d', '1d-seq-2d']:
                neighbour_x_positions = torch.tensor([], dtype=torch.int64)
                neighbour_y_positions = torch.tensor([], dtype=torch.int64)
            for i, neighbor_idx in enumerate(neighbor_indices):
                neighbour_line, neighbour_label, neighbour_x, neighbour_y = self.get_line_features_and_labels(neighbor_idx, idx)
                neighbour_lines = torch.cat((neighbour_lines, torch.tensor(neighbour_line).reshape(1, -1)))
                neighbour_labels = torch.cat((neighbour_labels, torch.tensor([neighbour_label], dtype=torch.float32)))
                if self.positional_encoding in ['2d', '1d-seq-2d']:
                    neighbour_x_positions = torch.cat((neighbour_x_positions, torch.tensor([neighbour_x])))
                    neighbour_y_positions = torch.cat((neighbour_y_positions, torch.tensor([neighbour_y])))

            input_tensor = torch.cat((neighbour_lines[:len(neighbour_lines)//2], input_tensor, neighbour_lines[len(neighbour_lines)//2:]))
            labels = labels.unsqueeze(0)
            labels = torch.cat((neighbour_labels[:len(neighbour_labels)//2, :], labels, neighbour_labels[len(neighbour_labels)//2:, :]))
            if self.positional_encoding in ['2d', '1d-seq-2d']:
                x_positions = torch.cat((neighbour_x_positions[:len(neighbour_x_positions)//2], x_positions, neighbour_x_positions[len(neighbour_x_positions)//2:]))
                y_positions = torch.cat((neighbour_y_positions[:len(neighbour_y_positions)//2], y_positions, neighbour_y_positions[len(neighbour_y_positions)//2:]))
            
        
        if self.positional_encoding == '1d-page':
            first_non_padding_position = -1
            last_non_padding_position = len(input_tensor) - 1
            for i, label in enumerate(labels):
                if label[-1] == 0 and first_non_padding_position == -1:
                    first_non_padding_position = i
                if label[-1] == 1 and first_non_padding_position != -1:
                    last_non_padding_position = i - 1
                    break
            first_non_padding_position_on_page = 0
            for i in range(idx - len(labels) // 2, idx, -1):
                if 0 > i >= len(self.data["lines"]):
                    break
                if self.data["lines"][i]["page_id"] != self.data["lines"][idx]["page_id"]:
                    break
                first_non_padding_position_on_page += 1
                
            positions = torch.full((input_tensor.shape[0], 1), -1)
            positions[self.neighbour_lines_cnt] = 0
            for i, j in zip(range(first_non_padding_position, last_non_padding_position + 1), range(last_non_padding_position - first_non_padding_position + 1)):
                positions[i] = first_non_padding_position_on_page + j
            input_tensor = torch.cat((input_tensor, positions), dim=1)
            
        if self.positional_encoding == '2d' or self.positional_encoding == '1d-seq-2d':
            input_tensor = torch.cat((input_tensor, x_positions.unsqueeze(1), y_positions.unsqueeze(1)), dim=1)        

        return {"input": input_tensor, "target": labels}
