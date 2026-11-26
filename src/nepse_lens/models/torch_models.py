"""Optional PyTorch models.

Torch is an extra (``pip install -e '.[torch]'``) so the core pipeline and CI
stay light. Everything here is guarded by ``HAS_TORCH`` and raises a clear
ImportError instead of failing at import time.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised only when torch is installed
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False


def make_windows(matrix: np.ndarray, seq_len: int) -> np.ndarray:
    """(T, F) -> (T - seq_len + 1, seq_len, F)."""
    if len(matrix) < seq_len:
        return np.empty((0, seq_len, matrix.shape[1]), dtype=np.float32)
    return np.stack([matrix[i : i + seq_len] for i in range(len(matrix) - seq_len + 1)])


if HAS_TORCH:

    class LSTMAutoencoder(nn.Module):
        def __init__(self, n_features: int, hidden: int = 32, latent: int = 8):
            super().__init__()
            self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
            self.to_latent = nn.Linear(hidden, latent)
            self.from_latent = nn.Linear(latent, hidden)
            self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
            self.out = nn.Linear(hidden, n_features)

        def forward(self, x):  # (B, T, F)
            _, (hidden, _) = self.encoder(x)
            latent = self.to_latent(hidden[-1])
            seed = self.from_latent(latent).unsqueeze(1).repeat(1, x.size(1), 1)
            decoded, _ = self.decoder(seed)
            return self.out(decoded)

    class TemporalBlock(nn.Module):
        def __init__(self, n_in: int, n_out: int, kernel_size: int, dilation: int, dropout: float = 0.2):
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.conv1 = nn.utils.parametrizations.weight_norm(
                nn.Conv1d(n_in, n_out, kernel_size, padding=padding, dilation=dilation)
            )
            self.conv2 = nn.utils.parametrizations.weight_norm(
                nn.Conv1d(n_out, n_out, kernel_size, padding=padding, dilation=dilation)
            )
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()

        def forward(self, x):
            out = self.dropout(self.relu(self.conv1(x)))
            out = self.dropout(self.relu(self.conv2(out)))
            out = out[..., -x.size(2) :]
            return self.relu(out + self.downsample(x))

    class TCNClassifier(nn.Module):
        def __init__(self, n_features: int, hidden: int = 32, dilations: tuple[int, ...] = (1, 2, 4)):
            super().__init__()
            blocks = []
            channels = n_features
            for dilation in dilations:
                blocks.append(TemporalBlock(channels, hidden, kernel_size=3, dilation=dilation))
                channels = hidden
            self.network = nn.Sequential(*blocks)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):  # (B, T, F)
            x = x.transpose(1, 2)
            encoded = self.network(x)
            pooled = encoded.mean(dim=2)
            return self.head(pooled).squeeze(-1)


def train_lstm_autoencoder(
    windows: np.ndarray,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 42,
):
    if not HAS_TORCH:
        raise ImportError("PyTorch is required: pip install -e '.[torch]'")
    torch.manual_seed(seed)
    model = LSTMAutoencoder(windows.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    data = torch.tensor(windows, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(data))
        for start in range(0, len(data), batch_size):
            batch = data[order[start : start + batch_size]]
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def reconstruction_scores(model, windows: np.ndarray, batch_size: int = 512) -> np.ndarray:
    if not HAS_TORCH:
        raise ImportError("PyTorch is required: pip install -e '.[torch]'")
    data = torch.tensor(windows, dtype=torch.float32)
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size]
            error = ((model(batch) - batch) ** 2).mean(dim=(1, 2))
            scores.append(error.numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


def train_tcn(
    windows: np.ndarray,
    labels: np.ndarray,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 42,
):
    if not HAS_TORCH:
        raise ImportError("PyTorch is required: pip install -e '.[torch]'")
    torch.manual_seed(seed)
    model = TCNClassifier(windows.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    data = torch.tensor(windows, dtype=torch.float32)
    target = torch.tensor(labels, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(data))
        for start in range(0, len(data), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(data[idx]), target[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict_tcn(model, windows: np.ndarray, batch_size: int = 512) -> np.ndarray:
    if not HAS_TORCH:
        raise ImportError("PyTorch is required: pip install -e '.[torch]'")
    data = torch.tensor(windows, dtype=torch.float32)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            logits = model(data[start : start + batch_size])
            outputs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
