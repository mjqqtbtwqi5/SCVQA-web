import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,
)

import math


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.module_list = nn.Sequential(
            *list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-2]
        )

        for p in self.module_list.parameters():
            p.requires_grad = False

    def global_std_pool2d(self, x: Tensor):
        x = torch.std(x.view(x.size()[0], x.size()[1], -1, 1), dim=2, keepdim=True)
        return x

    def forward(self, x):
        for i, module in enumerate(self.module_list):
            x = module(x)
            if i == (len(self.module_list) - 1):
                mean = nn.functional.adaptive_avg_pool2d(x, 1)
                std = self.global_std_pool2d(x)
                return mean, std


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()

        self.module_list = nn.Sequential(
            *list(resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).children())[:-2]
        )

        for p in self.module_list.parameters():
            p.requires_grad = False

    def global_std_pool2d(self, x: Tensor):
        x = torch.std(x.view(x.size()[0], x.size()[1], -1, 1), dim=2, keepdim=True)
        return x

    def forward(self, x):
        for i, module in enumerate(self.module_list):
            x = module(x)
            if i == (len(self.module_list) - 1):
                mean = nn.functional.adaptive_avg_pool2d(x, 1)
                std = self.global_std_pool2d(x)
                return mean, std


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.module_list = nn.Sequential(
            *list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).children())[:-2]
        )

        for p in self.module_list.parameters():
            p.requires_grad = False

    def global_std_pool2d(self, x: Tensor):
        x = torch.std(x.view(x.size()[0], x.size()[1], -1, 1), dim=2, keepdim=True)
        return x

    def forward(self, x):
        for i, module in enumerate(self.module_list):
            x = module(x)
            if i == (len(self.module_list) - 1):
                mean = nn.functional.adaptive_avg_pool2d(x, 1)
                std = self.global_std_pool2d(x)
                return mean, std


class ResNet101(nn.Module):
    def __init__(self):
        super().__init__()

        self.module_list = nn.Sequential(
            *list(resnet101(weights=ResNet101_Weights.IMAGENET1K_V2).children())[:-2]
        )

        for p in self.module_list.parameters():
            p.requires_grad = False

    def global_std_pool2d(self, x: Tensor):
        x = torch.std(x.view(x.size()[0], x.size()[1], -1, 1), dim=2, keepdim=True)
        return x

    def forward(self, x):
        for i, module in enumerate(self.module_list):
            x = module(x)
            if i == (len(self.module_list) - 1):
                mean = nn.functional.adaptive_avg_pool2d(x, 1)
                std = self.global_std_pool2d(x)
                return mean, std


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self,
        device: str,
        frame_size: int = 300,
        p_dropout: float = 0.5,
        feature_size: int = 4096,
        d_model=64,
        nhead=8,
        num_encoder_layers=8,
        dim_feedforward=32,
        dropout=0.1,
    ):
        super().__init__()

        self.device = device

        self.fc0 = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=d_model),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(in_features=d_model, out_features=d_model),
        )

        self.pos_encoder = PositionalEncoding(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self.fc1 = nn.Linear(d_model, 1)
        self.fc2 = nn.Linear(in_features=frame_size, out_features=1)

    def forward(self, x):

        x = self.fc0(x)

        x = self.pos_encoder(x)
        x = self.encoder(x)

        x = self.fc1(x)
        x = self.fc2(x.squeeze(dim=2))
        x = x.squeeze(dim=1)
        return x


class LSTM(nn.Module):
    def __init__(
        self,
        device: str,
        frame_size: int = 300,
        p_dropout: float = 0.5,
        feature_size: int = 4096,
        input_size: int = 64,
        hidden_size: int = 32,
        num_layers: int = 8,
    ):
        super().__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fc0 = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=input_size),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(in_features=input_size, out_features=input_size),
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc1 = nn.Linear(in_features=hidden_size, out_features=1)

        self.fc2 = nn.Linear(in_features=frame_size, out_features=1)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.fc0(x)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            device=self.device
        )
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            device=self.device
        )

        x, _ = self.lstm(x, (h0, c0))

        x = self.fc1(x)
        x = self.fc2(x.squeeze(dim=2))
        x = x.squeeze(dim=1)
        return x
