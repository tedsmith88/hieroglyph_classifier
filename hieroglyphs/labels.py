"""labels.csv: the class-index -> Gardiner-code mapping saved next to the weights."""
import csv
from pathlib import Path

from .config import WEIGHTS_DIR


def load_labels(path: Path = WEIGHTS_DIR / "labels.csv") -> list[str]:
    """Gardiner codes in class-index order."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda row: int(row["index"]))
    return [row["gardiner_code"] for row in rows]


def save_labels(classes: list[str], path: Path) -> None:
    """Write `classes` (ImageFolder order) as index,gardiner_code."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "gardiner_code"])
        writer.writerows(enumerate(classes))
