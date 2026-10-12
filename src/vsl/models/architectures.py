"""
Neural Network Architectures for Vietnamese Sign Language (VSL) Recognition.
Includes:
- VSL_BiGRU: Bidirectional GRU with Temporal Attention (Fastest, CPU < 3.5ms)
- VSL_BiLSTM: Bidirectional LSTM
- VSL_Transformer: Sequence Transformer Encoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vsl.config import settings, ACTIONS_METADATA


class TemporalAttention(nn.Module):
    """
    Self-attention layer to weight the importance of different frames
    in a dynamic sign gesture sequence.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, rnn_outputs: torch.Tensor):
        # rnn_outputs: (batch_size, seq_len, hidden_dim)
        scores = self.attention(rnn_outputs)  # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)    # (batch_size, seq_len, 1)
        context = torch.sum(weights * rnn_outputs, dim=1)  # (batch_size, hidden_dim)
        return context, weights


class VSL_BiGRU(nn.Module):
    """
    Bidirectional GRU Network for Real-Time Sign Language Recognition.
    - Low latency & minimal compute compared to heavy CNN-3D or Transformer.
    - Bi-directional temporal context captures gesture initiation and culmination.
    """
    def __init__(
        self,
        input_dim: int = settings.INPUT_FEATURE_DIM,
        hidden_dim: int = settings.HIDDEN_DIM,
        num_layers: int = settings.NUM_LAYERS,
        num_classes: int = len(ACTIONS_METADATA),
        dropout: float = settings.DROPOUT
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        gru_out_dim = hidden_dim * 2
        self.attention = TemporalAttention(gru_out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        projected = self.input_projection(x)
        gru_out, _ = self.gru(projected)
        context, _ = self.attention(gru_out)
        logits = self.classifier(context)
        return logits


class VSL_BiLSTM(nn.Module):
    """
    Bidirectional LSTM Network for Sign Language Recognition.
    """
    def __init__(
        self,
        input_dim: int = settings.INPUT_FEATURE_DIM,
        hidden_dim: int = settings.HIDDEN_DIM,
        num_layers: int = settings.NUM_LAYERS,
        num_classes: int = len(ACTIONS_METADATA),
        dropout: float = settings.DROPOUT
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        lstm_out_dim = hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, _) = self.lstm(x)
        h_concat = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        logits = self.fc(h_concat)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class VSL_Transformer(nn.Module):
    """
    Lightweight Temporal Transformer Encoder for Sign Action Classification.
    """
    def __init__(
        self,
        input_dim: int = settings.INPUT_FEATURE_DIM,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = len(ACTIONS_METADATA),
        dropout: float = 0.2
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        pooled = torch.mean(encoded, dim=1)
        logits = self.classifier(pooled)
        return logits
