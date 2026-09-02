"""Datasets and the train/val/test split.

Replaces the published notebook's `filenames.pth` + `test_1ch.csv` pair, which
hardcoded absolute paths into a dataset that is not distributed. Point
`--data-dir` at any directory laid out as:

    data/
      images/     strip photos      (RGB, any format PIL reads)
      masks/      binary masks      (same filename as the image)
      labels.csv  filename,label    (label: 0 = negative, 1 = positive)

"""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_labels(data_dir):
    """labels.csv -> {filename: int label}. Accepts an optional header row."""
    path = Path(data_dir) / "labels.csv"
    labels = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2 or not row[1].strip().lstrip("-").isdigit():
                continue  # header or blank line
            labels[row[0].strip()] = int(row[1])
    if not labels:
        raise ValueError(f"no usable rows in {path}")
    return labels


def list_pairs(data_dir, labels=None):
    """Return [(image_path, mask_path, label)], skipping images with no mask."""
    data_dir = Path(data_dir)
    labels = read_labels(data_dir) if labels is None else labels
    pairs = []
    for img in sorted((data_dir / "images").iterdir()):
        mask = data_dir / "masks" / img.name
        if not mask.exists():
            continue
        pairs.append((img, mask, labels.get(img.name)))
    if not pairs:
        raise ValueError(f"no image/mask pairs found under {data_dir}")
    return pairs


def load_mask(path):
    """Read a mask and binarise it at 128, as the published code did."""
    m = np.array(Image.open(path).convert("L"))
    return (m >= 128).astype(np.uint8)


class SegDataset(Dataset):
    """Photo -> binary mask. Photos are rescaled per-image to [0, 255]."""

    def __init__(self, pairs, transform=None):
        self._pairs = list(pairs)
        self._transform = transform

    def __getitem__(self, index):
        img_path, mask_path, _ = self._pairs[index]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        mm, MM = image.min(), image.max()
        image = (image - mm) / (MM - mm) * 255
        mask = load_mask(mask_path)

        if self._transform is not None:
            aug = self._transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        return image, mask

    def __len__(self):
        return len(self._pairs)


class ClassDataset(Dataset):
    """Mask -> positive/negative.

    The classifier's input is the mask, not the photo (see the published
    training loop). During training the mask is ground truth; at inference
    `src.predict` feeds it the segmentation network's predicted mask.
    """

    def __init__(self, pairs, transform=None):
        self._pairs = list(pairs)
        self._transform = transform

    def __getitem__(self, index):
        img_path, mask_path, label = self._pairs[index]
        if label is None:
            raise KeyError(f"{img_path.name} has no entry in labels.csv")
        mask = (load_mask(mask_path) * 255).astype(np.float32)  # 1-ch: classifier conv1 is 1-in

        if self._transform is not None:
            mask = self._transform(image=mask)["image"]
        return mask, torch.tensor([float(label)])

    def __len__(self):
        return len(self._pairs)


def dice_coef(y_true, y_pred, smooth=1e-4):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()
    y_true_f, y_pred_f = y_true.flatten(), y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
