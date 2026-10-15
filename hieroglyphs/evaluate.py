"""Accuracy measurement. Always runs in eval mode (BatchNorm must not update)."""
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader) -> float:
    """Top-1 accuracy of `model` over `loader`."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return correct / total
