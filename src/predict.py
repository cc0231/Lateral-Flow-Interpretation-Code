"""Inference: an LFA strip image -> positive / negative.

    image --[segmentation: unet | MnUV3 @ 256x256]--> band mask
    mask  --[classifier @ 256x256]-------------------> P(positive)

    python -m src.predict --image strip.jpg \
        --seg-weights weights/mnuv3_seg_train_no_test_no.pth \
        --cls-weights weights/classifier.pth \
        --arch mnuv3

Reference:
  Xue M, Gonzalez DH, Osikpa E, Gao X, Lillehoj PB. Sensors & Diagnostics, 2024.
  https://doi.org/10.1039/d4sd00314d
"""

import argparse
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from .models import MnUV3, Network, unet

SIZE = 256  # both nets: the segmentation input, and the classifier's baked-in FC size

transform = A.Compose([A.Normalize(mean=0.0, std=1.0), A.Resize(SIZE, SIZE), ToTensorV2()])

# Raw MnUV3 checkpoints name their blocks bneck2/bneck13 where models.py says
# bneck1/bneck2. Applied only when the old naming is actually present, so the
# remap is idempotent and safe on already-converted weights.
RENAME = {"bneck2.": "bneck1.", "bneck13.": "bneck2."}


def load_weights(model, path, device):
    """Load a checkpoint saved as {'state_dict': ...} or as a bare state_dict."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    needs_rename = any(k.startswith("bneck13.") for k in state)
    out = {}
    for k, v in state.items():
        k = k.replace("module.", "", 1)
        if needs_rename:
            for a, b in RENAME.items():
                if k.startswith(a):
                    k = b + k[len(a):]
                    break
        out[k] = v
    model.load_state_dict(out)
    model.to(device).eval()
    return model


def segment(image, model, device):
    """Strip image (HxWx3 uint8) -> band mask (256x256 uint8, values 0/1)."""
    x = image.astype(np.float32)
    mm, MM = x.min(), x.max()
    x = (x - mm) / (MM - mm) * 255  # per-image rescale, as in training

    x = transform(image=x)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def classify(mask, model, device):
    """Band mask -> P(positive). The classifier takes a 1-channel mask."""
    m = (mask * 255).astype(np.float32)
    x = transform(image=m)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        return float(model(x).reshape(-1)[0])


def predict(image_path, seg_model, cls_model, device, threshold=0.5):
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = segment(image, seg_model, device)
    prob = classify(mask, cls_model, device)
    return {
        "image": str(image_path),
        "probability_positive": prob,
        "call": "POSITIVE" if prob >= threshold else "NEGATIVE",
        "mask_coverage_px": int(mask.sum()),
        "mask": mask,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, nargs="+", help="one or more LFA strip images")
    ap.add_argument("--seg-weights", required=True)
    ap.add_argument("--cls-weights", required=True)
    ap.add_argument("--arch", choices=["unet", "mnuv3"], default="mnuv3")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save-mask", metavar="DIR", help="write predicted masks here as PNG")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    seg_model = load_weights(unet() if args.arch == "unet" else MnUV3(), args.seg_weights, device)
    cls_model = load_weights(Network(), args.cls_weights, device)

    if args.save_mask:
        Path(args.save_mask).mkdir(parents=True, exist_ok=True)

    print(f"{'image':<40} {'P(pos)':>8}  call")
    print("-" * 62)
    for img in args.image:
        if not Path(img).exists():
            print(f"{img:<40} {'--':>8}  FILE NOT FOUND", file=sys.stderr)
            continue
        r = predict(img, seg_model, cls_model, device, args.threshold)
        print(f"{Path(img).name:<40} {r['probability_positive']:>8.3f}  {r['call']}")
        if args.save_mask:
            Image.fromarray(r["mask"] * 255).save(Path(args.save_mask) / f"{Path(img).stem}_mask.png")


if __name__ == "__main__":
    main()
