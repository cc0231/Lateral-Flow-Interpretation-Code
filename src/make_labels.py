"""Derive positive/negative labels from segmentation masks.
    python -m src.make_labels --data-dir data
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

STRUCT = np.ones((3, 3), int)  # 8-connectivity, matching bwconncomp(img, 8)


def count_bands(mask_path):
    m = (np.asarray(Image.open(mask_path).convert("L")) >= 128).astype(int)
    return ndimage.label(m, structure=STRUCT)[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=None, help="default: <data-dir>/labels.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out) if args.out else data_dir / "labels.csv"

    rows, ambiguous = [], []
    for mp in sorted((data_dir / "masks").iterdir()):
        n = count_bands(mp)
        if n in (1, 2):
            rows.append((mp.name, n - 1))
        else:
            ambiguous.append((mp.name, n))

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "label"])
        w.writerows(rows)

    pos = sum(l for _, l in rows)
    print(f"{len(rows)} labelled ({pos} positive / {len(rows) - pos} negative) -> {out}")
    if ambiguous:
        print(f"{len(ambiguous)} ambiguous, left out:")
        for name, n in ambiguous[:10]:
            print(f"    {name}  ({n} components)")


if __name__ == "__main__":
    main()
