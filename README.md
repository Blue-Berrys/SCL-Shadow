# SCL-Shadow

Official code for **"Self-Calibrated Shadow Detection with Spatial Consistency Constraints under Noisy Labels"** (submitted to *The Visual Computer*).

A training-time, backbone-agnostic toolkit that makes shadow detectors robust to label noise — **without changing the raw labels and with zero extra inference cost**. It builds on SDDNet and adds three training-only components:

- **ICA** — Iterative Confidence Aggregation: per-image EMA prediction history → down-weights temporally unreliable pixels.
- **SCV** — Spatial Consistency Verification: local boundary-structure agreement → down-weights spatially inconsistent regions.
- **BRLF** — Boundary Refined Loss Function: warm-up + per-epoch alternation between the original and reliability-weighted loss.

![Framework](assets/overview.png)

## Install

```bash
pip install -r requirements.txt
```

## Data

Arrange each split as paired images and masks matched by filename stem; see [`tools/prepare_data.md`](tools/prepare_data.md).

```
<root>/images/  0001.jpg ...
<root>/masks/   0001.png ...
```

## Train / Test

```bash
python tools/train.py --config configs/sbu.yaml
python tools/test.py  --config configs/ucf_eval.yaml --checkpoint checkpoints/scl_sbu_best.pth --root data/UCF/test
python tools/infer_video.py --config configs/visha_eval.yaml --checkpoint checkpoints/scl_sbu_best.pth --frames data/ViSha/video1/frames --out outputs/video1
```

Hyperparameters (paper defaults) are in `configs/*.yaml`. The paper uses SDDNet as the backbone; wrap the official [SDDNet](https://github.com/rmcong/SDDNet_ACMMM23) via `scl.backbone.SDDNetAdapter` (a small reference U-Net is included so the pipeline runs out of the box).

## Results (BER ↓)

| Method | ISTD | SBU | UCF |
|--------|:----:|:---:|:---:|
| SDDNet (baseline) | 1.27 | 2.94 | 6.59 |
| **Ours** | 1.14 | **2.70** | **6.20** |

## Citation

```bibtex
@article{xie2026sclshadow,
  title   = {Self-Calibrated Shadow Detection with Spatial Consistency Constraints under Noisy Labels},
  author  = {Xie, Jiaxuan and Chen, Xiao-Diao and Mo, Yuchang and Wu, Wen},
  journal = {The Visual Computer},
  year    = {2026}
}
```

Released under the [MIT License](LICENSE).
