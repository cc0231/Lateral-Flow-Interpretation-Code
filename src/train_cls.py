"""Train the classifier (mask -> positive / negative).
    python -m src.train_cls --data-dir data --epochs 300
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from .data import ClassDataset, list_pairs
from .models import Network
from .split import load_split
from .train_seg import subset

CLS_SIZE = 256  # Network's FC input size is fixed at this resolution


def transforms(augment):
    ops = [A.Normalize(mean=0.0, std=1.0)]
    if augment:
        ops += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Rotate(limit=10, p=0.5)]
    return A.Compose(ops + [A.Resize(CLS_SIZE, CLS_SIZE), ToTensorV2()])


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Returns (accuracy, mean loss). Accuracy is 1 - mean|Y - p|, as published."""
    train = optimizer is not None
    model.train() if train else model.eval()
    total_err = total_loss = n = 0

    with torch.set_grad_enabled(train):
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            R = model(X).reshape(-1, 1)
            L = loss_fn(R.float(), Y.float())
            if train:
                optimizer.zero_grad()
                L.backward()
                optimizer.step()
            total_err += float(torch.abs(Y - R.data).sum())
            total_loss += L.data.item() * X.shape[0]
            n += X.shape[0]
    return 1 - total_err / n, total_loss / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="splits/split.json")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-late", type=float, default=1e-4)
    ap.add_argument("--lr-drop-epoch", type=int, default=20)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--augment-eval", action="store_true",
                    help="augment during validation as the published notebook did")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    pairs = list_pairs(args.data_dir)
    split = load_split(args.split)
    train_pairs, val_pairs = subset(pairs, split["train"]), subset(pairs, split["val"])
    missing = sum(1 for p in train_pairs + val_pairs if p[2] is None)
    if missing:
        raise SystemExit(f"{missing} images have no label in labels.csv -- cannot train the classifier")
    print(f"train {len(train_pairs)} | val {len(val_pairs)}")

    train_dl = DataLoader(ClassDataset(train_pairs, transforms(True)), batch_size=args.batch_size,
                          num_workers=args.workers, shuffle=True)
    val_dl = DataLoader(ClassDataset(val_pairs, transforms(args.augment_eval)), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False)

    for trial in range(args.trials):
        torch.manual_seed(args.seed + trial)
        np.random.seed(args.seed + trial)
        torch.cuda.empty_cache()

        model = Network().to(device)
        opt_early = torch.optim.Adam(model.parameters(), lr=args.lr)
        opt_late = torch.optim.Adam(model.parameters(), lr=args.lr_late)
        loss_fn = torch.nn.BCELoss()
        history = {k: np.zeros(args.epochs) for k in ("train_acc", "train_loss", "val_acc", "val_loss")}

        stamp = datetime.now().strftime("%y%m%d_%H%M")
        run_dir = Path(args.out) / (f"{stamp}_epoch{args.epochs}_batch{args.batch_size}"
                                    f"_img{CLS_SIZE}_adam{args.lr}_classify_trial{trial}")
        run_dir.mkdir(parents=True, exist_ok=True)
        print(run_dir)
        (run_dir / "config.json").write_text(json.dumps(vars(args) | {"trial": trial}, indent=1))

        for epoch in range(1, args.epochs):
            optimizer = opt_late if epoch > args.lr_drop_epoch else opt_early
            print("=" * 30, f"\nEpoch {epoch} / {args.epochs}")
            history["train_acc"][epoch], history["train_loss"][epoch] = run_epoch(
                model, train_dl, loss_fn, device, optimizer)
            print(f"Loss: {history['train_loss'][epoch]:3.3f}, Accuracy: {history['train_acc'][epoch]:3.3f}")
            history["val_acc"][epoch], history["val_loss"][epoch] = run_epoch(model, val_dl, loss_fn, device)
            print(f"Validation Loss: {history['val_loss'][epoch]:3.3f}, Accuracy: {history['val_acc'][epoch]:3.3f}")

            last = epoch == args.epochs - 1
            if (epoch % 10 == 9 and epoch > 70) or last:
                ck = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                if last:
                    ck |= {k: v for k, v in history.items()}
                torch.save(ck, run_dir / f"{epoch}.pth")


if __name__ == "__main__":
    main()
