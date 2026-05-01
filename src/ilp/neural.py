"""Neural scoring models used by the iLP ligand preparation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


class MolAIEncoder(nn.Module):
    """MolAI SMILES encoder used to embed protonation-state SMILES."""

    def __init__(self, input_dim: int = 34, hidden_dim: int = 1024, latent_dim: int = 512):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.latent = nn.Linear(hidden_dim * 6, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, (h1, c1) = self.lstm1(x)
        out2, (h2, c2) = self.lstm2(out1)
        _, (h3, c3) = self.lstm3(out2)
        states = torch.cat(
            [
                h1.squeeze(0),
                c1.squeeze(0),
                h2.squeeze(0),
                c2.squeeze(0),
                h3.squeeze(0),
                c3.squeeze(0),
            ],
            dim=1,
        )
        return torch.tanh(self.latent(states))


class ILPClassifier(nn.Module):
    """One iLP classifier head in the protonation-state ensemble."""

    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.dense1 = nn.Linear(latent_dim, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 512)
        self.output = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        return torch.sigmoid(self.output(x))


class ILPBundle(nn.Module):
    """MolAI encoder plus the five-model iLP ensemble."""

    def __init__(self, ensemble_size: int = 5):
        super().__init__()
        self.encoder = MolAIEncoder()
        self.classifiers = nn.ModuleList([ILPClassifier() for _ in range(ensemble_size)])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def predict_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        predictions = [classifier(latent).squeeze(1) for classifier in self.classifiers]
        return torch.stack(predictions, dim=0).mean(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_from_latent(self.encode(x))


def choose_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch does not detect a CUDA device.")
    return torch.device(requested_device)


def load_tokenizer(model_dir: Path) -> tuple[dict[str, int], dict[int, str]]:
    tokenizer_path = model_dir / "tokenizer.json"
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    char_to_int = {str(key): int(value) for key, value in data["char_to_int"].items()}
    int_to_char = {int(key): str(value) for key, value in data["int_to_char"].items()}
    return char_to_int, int_to_char


def load_bundle(
    model_dir: str | Path,
    requested_device: str = "auto",
) -> tuple[ILPBundle, dict[str, int], dict[int, str], torch.device, dict[str, Any]]:
    model_dir = Path(model_dir)
    device = choose_device(requested_device)
    char_to_int, int_to_char = load_tokenizer(model_dir)

    try:
        checkpoint = torch.load(
            model_dir / "ilp_model.pt",
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(model_dir / "ilp_model.pt", map_location=device)
    metadata = checkpoint.get("metadata", {})
    bundle = ILPBundle(ensemble_size=int(metadata.get("ensemble_size", 5)))
    bundle.load_state_dict(checkpoint["state_dict"])
    bundle.to(device)
    bundle.eval()
    return bundle, char_to_int, int_to_char, device, metadata
