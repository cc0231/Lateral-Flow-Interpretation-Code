"""Evaluate the end-to-end pipeline on a split and print the paper's metrics.

    python -m src.evaluate --data-dir data --split splits/split.json --subset test \
        --seg-weights weights/mnuv3_seg_train_no_test_no.pth \
        --cls-weights weights/classifier.pth --arch mnuv3

Reports accuracy, sensitivity, specificity and mean segmentation Dice. Numbers
come only from what is actually computed here -- nothing is carried over from
the paper.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import dice_coef, list_pairs, load_mask
from .models import MnUV3, Network, unet
from .predict import load_weights, predict
from .split import load_split


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    safe = lambda a, b: a / b if b else float("nan")
    return {
        "n": len(y_true), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": safe(tp + tn, len(y_true)),
        "sensitivity": safe(tp, tp + fn),
        "specificity": safe(tn, tn + fp),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default=None, help="omit to evaluate every image in --data-dir")
    ap.add_argument("--subset", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seg-weights", required=True)
    ap.add_argument("--cls-weights", required=True)
    ap.add_argument("--arch", choices=["unet", "mnuv3"], default="mnuv3")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-csv", help="write per-case results here")
    args = ap.parse_args()

    device = torch.device(args.device)
    seg = load_weights(unet() if args.arch == "unet" else MnUV3(), args.seg_weights, device)
    cls = load_weights(Network(), args.cls_weights, device)

    pairs = list_pairs(args.data_dir)
    if args.split:
        keep = set(load_split(args.split)[args.subset])
        pairs = [p for p in pairs if p[0].name in keep]
    pairs = [p for p in pairs if p[2] is not None]
    if not pairs:
        raise SystemExit("nothing to evaluate: no labelled images in this subset")

    y_true, y_pred, probs, dices, names = [], [], [], [], []
    for img_path, mask_path, label in pairs:
        r = predict(img_path, seg, cls, device, args.threshold)
        gt = load_mask(mask_path).astype(np.float32)
        pm = np.asarray(Image.fromarray(r["mask"] * 255).resize(gt.shape[::-1])) >= 128
        names.append(img_path.name)
        y_true.append(label)
        y_pred.append(r["probability_positive"] >= args.threshold)
        probs.append(r["probability_positive"])
        dices.append(dice_coef(gt, pm.astype(np.float32)))

    m = metrics(y_true, y_pred)
    print(f"\n{args.arch}  |  {m['n']} images"
          f"{'  (' + args.subset + ' split)' if args.split else ''}")
    print(f"  accuracy     {m['accuracy']:6.1%}")
    print(f"  sensitivity  {m['sensitivity']:6.1%}")
    print(f"  specificity  {m['specificity']:6.1%}")
    print(f"  mean Dice    {np.mean(dices):6.4f}")
    print(f"  confusion    tp={m['tp']} tn={m['tn']} fp={m['fp']} fn={m['fn']}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["filename", "label", "predicted", "probability_positive", "dice"])
            for row in zip(names, y_true, y_pred, probs, dices):
                w.writerow([row[0], int(row[1]), int(row[2]), f"{row[3]:.4f}", f"{row[4]:.4f}"])
        print(f"\nper-case results -> {args.out_csv}")


if __name__ == "__main__":
    main()
