"""Final pipeline: train, save weights.pt + labels.csv, then report accuracy.

    ./venv/bin/python train.py --overwrite              # rebuild the canonical model
    ./venv/bin/python train.py --epochs 1 --out-dir runs/smoke
"""
import argparse
from pathlib import Path

from hieroglyphs import config
from hieroglyphs.data import HieroglyphTestDataset, make_loader, train_folder
from hieroglyphs.evaluate import evaluate
from hieroglyphs.training import save_checkpoint, train_model
from hieroglyphs.transforms import test_transform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--lr-fc", type=float, default=config.LR_FC)
    p.add_argument("--lr-layer4", type=float, default=config.LR_LAYER4)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--out-dir", type=Path, default=config.WEIGHTS_DIR)
    p.add_argument("--overwrite", action="store_true", help="allow replacing an existing weights.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if (args.out_dir / "weights.pt").exists() and not args.overwrite:
        raise SystemExit(f"{args.out_dir / 'weights.pt'} exists; pass --overwrite to replace it.")

    model, classes = train_model(
        seed=args.seed, epochs=args.epochs, weight_decay=args.weight_decay,
        lr_fc=args.lr_fc, lr_layer4=args.lr_layer4, verbose=True,
    )
    save_checkpoint(model, classes, args.out_dir)  # saved before evaluating, so a crash loses nothing
    print(f"Saved weights.pt and labels.csv to {args.out_dir}")

    train_set = train_folder(test_transform)  # clean (non-augmented) view of the training data
    test_set = HieroglyphTestDataset(train_set.class_to_idx, transform=test_transform)
    train_acc = evaluate(model, make_loader(train_set))
    test_acc = evaluate(model, make_loader(test_set))
    print(f"train={train_acc:.4f} test={test_acc:.4f}")


if __name__ == "__main__":
    main()
