"""Create a reproducible train/val/test split.
    python -m src.split --data-dir data --seed 42
"""

import argparse
import json
import random
from pathlib import Path

from .data import list_pairs


def make_split(data_dir, seed=42, test_frac=0.2, val_frac=0.3):
    """Split image filenames into test / train / val.

    `val_frac` is taken from the non-test remainder, matching the published
    `test_size=0.3` on the train/val call.
    """
    names = [img.name for img, _, _ in list_pairs(data_dir)]
    rng = random.Random(seed)
    rng.shuffle(names)

    n_test = int(round(len(names) * test_frac))
    test, remainder = names[:n_test], names[n_test:]
    n_val = int(round(len(remainder) * val_frac))
    val, train = remainder[:n_val], remainder[n_val:]
    return {"seed": seed, "train": sorted(train), "val": sorted(val), "test": sorted(test)}


def load_split(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="splits/split.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.3)
    args = ap.parse_args()

    split = make_split(args.data_dir, args.seed, args.test_frac, args.val_frac)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=1))
    print(f"train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])}  -> {out}")


if __name__ == "__main__":
    main()
