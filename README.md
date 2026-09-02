# [Rapid and automated interpretation of CRISPR-Cas13-based lateral flow assay test results using machine learning](https://pubs.rsc.org/sd/article-abstract/4/2/171/870940)

📄 **Paper:** [Xue *et al.*, *Sensors & Diagnostics*, 2024](https://doi.org/10.1039/d4sd00314d) (open access)

*Obtain rapid and accurate lateral flow assay (LFA) strips from ordinary smartphone photos with light-weight mobile-compatible models.*

**Contact [Mengyuan Xue](mailto:mengyuan.xue@utsouthwestern.edu) (mengyuan.xue@utsouthwestern.edu) or [Dr. Peter Lillehoj ](mailto:lillehoj@rice.edu)(lillehoj@rice.edu) for any questions or requests.**

```
image ──▶ segmentation (U-Net | MnUV3 @ 256²) ──▶ mask ──▶ classifier (@ 256²) ──▶ POSITIVE / NEGATIVE
```

## Input

A square image of the strip. Size does not matter as it is resized to 256x256 either way. The segmentation net finds the bands in whatever you give it. The performance will be the best when the size is a multiple of 256.

The classifier reads the **segmentation mask** in the second stage and subsequntly gives a binary prediction of the input strip photo.

## Results

Evaluated on held-out validation set: 3,146 unseen cropped images (1,569 negative / 1,577 positive)

| Model                                                 |         Params |         Accuracy | Sensitivity |      Specificity |
| ----------------------------------------------------- | -------------: | ---------------: | ----------: | ---------------: |
| U-Net                                                 |           30 M |           96.4 % |      96.0 % |           96.8 % |
| **MnUV3** (MobileNetV3 encoder + U-Net decoder) | **18 M** | **96.5 %** |      94.8 % | **98.3 %** |

Inference takes ~0.2 s per image.

**Try it without installing anything:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cc0231/Lateral-Flow-Interpretation-Code/blob/main/notebooks/demo.ipynb)

## Install

```bash
git clone https://github.com/cc0231/Lateral-Flow-Interpretation-Code
cd Lateral-Flow-Interpretation-Code
pip install -r requirements.txt
```

## Read a strip

Download `mnuv3_seg.pth` and `classifier.pth` from
[Releases](https://github.com/cc0231/Lateral-Flow-Interpretation-Code/releases)
into `weights/`, then:

```bash
python -m src.predict \
    --image sample_data/images/*.jpg \
    --seg-weights weights/mnuv3_seg.pth \
    --cls-weights weights/classifier.pth \
    --arch mnuv3 \
    --save-mask out/
```

```
image                                      P(pos)  call
--------------------------------------------------------------
strip_001.jpg                               0.981  POSITIVE
strip_002.jpg                               0.017  NEGATIVE
```

Pass `--arch unet --seg-weights weights/unet_seg.pth` for the larger model.

## Measure it

```bash
python -m src.evaluate --data-dir data --split splits/split.json --subset test     --seg-weights weights/mnuv3_seg_train_no_test_no.pth     --cls-weights weights/classifier.pth --arch mnuv3 --out-csv results.csv
```

Prints accuracy, sensitivity, specificity and mean Dice, and writes a per-case
CSV. Everything it reports is computed from the run, not copied from the paper.

## Train on your own strips

Lay your data out like this:

```
data/
  images/     strip photos
  masks/      binary masks, same filenames as the images
  labels.csv  filename,label      (0 = negative, 1 = positive)
```

`labels.csv` can be derived from the masks and 0 indicates a negative result, 1 indicates a positive result.

```bash
python -m src.make_labels --data-dir data
```

```bash
python -m src.split --data-dir data --seed 42   # writes splits/split.json
python -m src.train_seg   --data-dir data --arch mnuv3
python -m src.train_cls   --data-dir data
```

The split is seeded and written to disk, so a run can be repeated exactly.

## Released weights

| File                                |   Size |
| ----------------------------------- | -----: |
| `mnuv3_seg_train_no_test_no.pth`  |  83 MB |
| `mnuv3_seg_train_yes_test_no.pth` |  83 MB |
| `unet_seg_train_no_test_no.pth`   | 124 MB |
| `unet_seg_train_yes_test_no.pth`  | 124 MB |
| `classifier.pth`                  |  26 MB |

`train_yes/no` refers to augmentation during training and evaluation. The three
configurations score on *different* evaluation transforms and are not comparable
to each other — compare runs only within a configuration.

Checkpoints are weights-only (`state_dict` plus the training history) and load
under `torch.load(..., weights_only=True)`.

## Data

The full corpus is 637 device photos → 8,125 image/label pairs (4,253 negative,
3,872 positive), captured on an iPhone 13 and a Samsung Galaxy A52 under varied
lighting and backgrounds. It is **available from the authors on reasonable
request**; see the paper's data availability statement.

`sample_data/` holds 30 example crops (15 positive / 15 negative) with their
masks, so the inference command above runs out of the box. It is a
**demonstration sample, not an evaluation set**, so the accuracies in the table
cannot be reproduced from it.

## Citation

```bibtex
@article{xue2024lfa,
  title   = {Rapid and automated interpretation of {CRISPR-Cas13}-based lateral
             flow assay test results using machine learning},
  author  = {Xue, Mengyuan and Gonzalez, Diego H. and Osikpa, Emmanuel and
             Gao, Xue and Lillehoj, Peter B.},
  journal = {Sensors \& Diagnostics},
  year    = {2024},
  doi     = {10.1039/d4sd00314d}
}
```

## License

Code released under the MIT License (see `LICENSE`). Trained weights are
released under CC-BY-4.0. The paper is open access under CC-BY.

## Funding

National Institutes of Health (R61AI167037) and National Science Foundation
(CBET2431019 to X.G.). We thank Weinan Wang for assistance with data acquisition.