"""Train the segmentation network (unet or MnUV3).
    python -m src.train_seg --data-dir data --arch mnuv3 --epochs 300
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

from .data import SegDataset, dice_coef, list_pairs
from .models import MnUV3, unet
from .split import load_split


def transforms(size, augment):
    ops = [A.Normalize(mean=0.0, std=1.0)]
    if augment:
        ops += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Rotate(limit=10, p=0.5)]
    return A.Compose(ops + [A.Resize(size, size), ToTensorV2()])


def subset(pairs, names):
    keep = set(names)
    return [p for p in pairs if p[0].name in keep]


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass. Returns (mean Dice, mean loss). Training if optimizer is given."""
    train = optimizer is not None
    model.train() if train else model.eval()
    total_acc = total_loss = n = 0

    with torch.set_grad_enabled(train):
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            R = model(X)
            L = loss_fn(R, Y.long())
            if train:
                optimizer.zero_grad()
                L.backward()
                optimizer.step()
            pred = R.data.max(1)[1]
            total_acc += dice_coef(pred, Y) * X.shape[0]
            total_loss += L.data.item() * X.shape[0]
            n += X.shape[0]
    return total_acc / n, total_loss / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="splits/split.json")
    ap.add_argument("--arch", choices=["unet", "mnuv3"], default="mnuv3")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--augment-eval", action="store_true",
                    help="apply train-time augmentation during validation, as the published "
                         "notebook did; makes validation Dice non-deterministic")
    ap.add_argument("--resume", metavar="CKPT")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    pairs = list_pairs(args.data_dir)
    split = load_split(args.split)
    train_pairs, val_pairs = subset(pairs, split["train"]), subset(pairs, split["val"])
    print(f"train {len(train_pairs)} | val {len(val_pairs)}")

    train_ds = SegDataset(train_pairs, transforms(args.img_size, True))
    val_ds = SegDataset(val_pairs, transforms(args.img_size, args.augment_eval))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    for trial in range(args.trials):
        torch.manual_seed(args.seed + trial)
        np.random.seed(args.seed + trial)
        torch.cuda.empty_cache()

        model = (unet() if args.arch == "unet" else MnUV3()).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        start_epoch = 1

        history = {k: np.zeros(args.epochs) for k in ("train_acc", "train_loss", "val_acc", "val_loss")}

        if args.resume:
            ck = torch.load(args.resume, map_location=device)
            model.load_state_dict(ck["state_dict"])
            optimizer.load_state_dict(ck["optimizer"])
            start_epoch = ck["epoch"] + 1
            for k in history:
                if k in ck:
                    history[k][: len(ck[k])] = ck[k]
            run_dir = Path(args.resume).parent
        else:
            stamp = datetime.now().strftime("%y%m%d_%H%M")
            run_dir = Path(args.out) / (f"{stamp}_epoch{args.epochs}_batch{args.batch_size}"
                                        f"_img{args.img_size}_adam{args.lr}_{args.arch}_trial{trial}")
            run_dir.mkdir(parents=True, exist_ok=True)
        print(run_dir)
        (run_dir / "config.json").write_text(json.dumps(vars(args) | {"trial": trial}, indent=1))

        for epoch in range(start_epoch, args.epochs):
            print("=" * 30, f"\nEpoch {epoch} / {args.epochs}")
            history["train_acc"][epoch], history["train_loss"][epoch] = run_epoch(
                model, train_dl, loss_fn, device, optimizer)
            print(f"Loss: {history['train_loss'][epoch]:3.3f}, Dice: {history['train_acc'][epoch]:3.3f}")
            history["val_acc"][epoch], history["val_loss"][epoch] = run_epoch(model, val_dl, loss_fn, device)
            print(f"Validation Loss: {history['val_loss'][epoch]:3.3f}, Dice: {history['val_acc'][epoch]:3.3f}")

            last = epoch == args.epochs - 1
            if (epoch % 10 == 9 and epoch > 70) or last:
                ck = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                if last:
                    ck |= {k: v for k, v in history.items()}
                torch.save(ck, run_dir / f"{epoch}.pth")


if __name__ == "__main__":
    main()
