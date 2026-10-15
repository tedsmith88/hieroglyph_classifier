"""Datasets, loaders and the stratified dev split."""
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from .config import BATCH_SIZE, DEV_FRACTION, DEV_SEED, TEST_DIR, TRAIN_DIR


class HieroglyphTestDataset(Dataset):
    """Flat test folder with labels in the filename (e.g. 070048_F34.png -> F34).

    Files whose label is not a training class (UNKNOWN, M20) are skipped, so the
    label indices always agree with the training set.
    """

    def __init__(self, class_to_idx: dict[str, int], root_dir: Path = TEST_DIR,
                 transform: Callable | None = None):
        self.root_dir = Path(root_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.filenames = sorted(
            f.name for f in self.root_dir.glob("*.png")
            if self.label_of(f.name) in class_to_idx
        )

    @staticmethod
    def label_of(filename: str) -> str:
        """Gardiner code encoded in a test filename."""
        return filename.split("_")[1].removesuffix(".png")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        img = Image.open(self.root_dir / filename)
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[self.label_of(filename)]


def train_folder(transform: Callable) -> ImageFolder:
    """Training images, one folder per class; class index = sorted folder order."""
    return ImageFolder(root=str(TRAIN_DIR), transform=transform)


def make_loader(dataset: Dataset, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def dev_split(targets: list[int], fraction: float = DEV_FRACTION,
              seed: int = DEV_SEED) -> tuple[list[int], list[int]]:
    """Per-class split of training indices into (train, dev).

    Classes with a single image stay entirely in train (sklearn's stratify would fail).
    """
    by_class = defaultdict(list)
    for idx, label in enumerate(targets):
        by_class[label].append(idx)

    train_idx, dev_idx = [], []
    for indices in by_class.values():
        random.Random(seed).shuffle(indices)
        if len(indices) < 2:
            train_idx.extend(indices)
        else:
            n_dev = max(1, int(len(indices) * fraction))
            dev_idx.extend(indices[:n_dev])
            train_idx.extend(indices[n_dev:])
    return train_idx, dev_idx
