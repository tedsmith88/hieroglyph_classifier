"""The classifier: ResNet-18, frozen backbone, trainable layer4, fresh 170-way head."""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .config import NUM_CLASSES


def build_model(pretrained: bool = True) -> nn.Module:
    """ResNet-18 ready for fine-tuning. Use pretrained=False when loading saved weights."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(512, NUM_CLASSES)
    return model


def load_weights(model: nn.Module, path: Path) -> nn.Module:
    """Load a saved state_dict into `model` and switch it to eval mode."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()
