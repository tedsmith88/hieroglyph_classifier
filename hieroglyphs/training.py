"""Training, shared by the notebook experiments and by train.py (the final pipeline)."""
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import Subset

from .config import EPOCHS, LR_FC, LR_LAYER4, SEED, WEIGHT_DECAY
from .data import make_loader, train_folder
from .labels import save_labels
from .model import build_model
from .transforms import train_transform


def train_model(seed: int = SEED, transform: Callable = train_transform,
                weight_decay: float = WEIGHT_DECAY, epochs: int = EPOCHS,
                lr_fc: float = LR_FC, lr_layer4: float = LR_LAYER4,
                subset_idx: list[int] | None = None,
                verbose: bool = False) -> tuple[nn.Module, list[str]]:
    """Fine-tune from scratch and return (model in eval mode, class names).

    `subset_idx` restricts training to those indices (the dev-split experiments);
    None trains on the full training set. The seed is set first, so a run is reproducible.
    """
    torch.manual_seed(seed)
    dataset = train_folder(transform)
    train_set = dataset if subset_idx is None else Subset(dataset, subset_idx)
    loader = make_loader(train_set, shuffle=True)

    model = build_model(pretrained=True)
    optimizer = torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": lr_fc},
        {"params": model.layer4.parameters(), "lr": lr_layer4, "weight_decay": weight_decay},
    ])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for images, labels in loader:
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, last-batch loss: {loss.item():.4f}")

    return model.eval(), dataset.classes


def save_checkpoint(model: nn.Module, classes: list[str], out_dir: Path) -> None:
    """Write weights.pt and labels.csv into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "weights.pt")
    save_labels(classes, out_dir / "labels.csv")
