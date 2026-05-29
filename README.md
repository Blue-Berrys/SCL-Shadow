# SCL-Shadow

**Self-Calibrated Shadow Detection with Spatial Consistency Constraints under Noisy Labels**

Jiaxuan Xie, Xiao-Diao Chen, Yuchang Mo, Wen Wu — *submitted to The Visual Computer*

This repository provides the official training-time toolkit for SCL-Shadow: a
**backbone-agnostic** framework that makes shadow detectors robust to annotation
noise and spatially fragmented predictions **without modifying the raw labels and
without adding any inference-time cost**. The deployed network is identical to the
chosen backbone (the paper uses [SDDNet](#using-the-sddnet-backbone)).

The framework formulates noisy-label shadow detection as a *reliability-weighted
dense prediction* problem and estimates per-pixel supervision reliability from
two complementary views, fused by a periodic training schedule:

- **ICA — Iterative Confidence Aggregation** (temporal reliability): accumulates
  each image's prediction history via an EMA and down-weights temporally
  unstable / persistently-disagreeing pixels.
- **SCV — Spatial Consistency Verification** (spatial reliability): compares the
  local boundary structure of the prediction and the annotation and down-weights
  spatially inconsistent blocks.
- **BRLF — Boundary Refined Loss Function** (training schedule): after a warm-up,
  alternates per epoch between the original loss and the reliability-weighted
  loss to balance generalisation and stability.

---

## Highlights

- 🔌 **Plug-in, backbone-agnostic** — wrap any shadow detector that exposes a
  shadow map plus reconstruction / auxiliary losses.
- 🏷️ **Label-noise robust** — never edits the raw annotations; only re-weights
  per-pixel supervision.
- ⚡ **Zero inference overhead** — ICA/SCV/BRLF are training-only; inference is
  just the backbone.
- 🧩 **Faithful & tested** — modules implement the paper's Eqs. (4)–(11); unit
  tests check them against the equations, and a synthetic smoke test exercises
  the full loop with no datasets.

## Framework

![Framework Overview](assets/overview.png)

| ICA | SCV |
|-----|-----|
| ![ICA](assets/ica.png) | ![SCV](assets/scv.png) |

## Repository layout

```
scl/
  calibration/
    ica.py      # IterativeConfidenceAggregation  (Eqs. 4-6)
    scv.py      # SpatialConsistencyVerification   (Eqs. 7-8, fully vectorised)
    brlf.py     # BoundaryRefinedLoss schedule     (Eq. 11)
    __init__.py # combined_weight (gate fusion, Eq. 9)
  losses.py     # weighted-BCE assembly            (Eqs. 3, 10)
  metrics.py    # Balanced Error Rate (BER)
  backbone.py   # backbone interface + ReferenceUNet + SDDNetAdapter
  data/         # ShadowDataset, joint transforms
  engine/       # Trainer (BRLF loop) + evaluator
  utils/        # config, seeding, PolyLR, checkpoints, logging
tools/          # train.py, test.py, infer_video.py, smoke_test.py, prepare_data.md
configs/        # sbu.yaml, istd.yaml, ucf_eval.yaml, visha_eval.yaml
tests/          # unit tests for ICA / SCV / BRLF
```

## Installation

```bash
git clone https://github.com/Blue-Berrys/SCL-Shadow.git
cd SCL-Shadow
pip install -r requirements.txt
```

## Quick start (no datasets needed)

Run the end-to-end smoke test on synthetic data to verify the install:

```bash
python tools/smoke_test.py
```

Run the unit tests (checks the modules against the paper's equations):

```bash
pip install pytest && pytest -q
```

## Data preparation

Arrange each split as paired images and masks matched by filename stem:

```
<root>/
    images/  0001.jpg 0002.jpg ...
    masks/   0001.png 0002.png ...
```

See [`tools/prepare_data.md`](tools/prepare_data.md) for SBU / ISTD / UCF / ViSha
details. All inputs are resized to `416×416` (must be even for SCV's `2×2`
blocking).

## Training & evaluation

```bash
# Train on SBU (edit data roots in the config first)
python tools/train.py --config configs/sbu.yaml

# Train on ISTD
python tools/train.py --config configs/istd.yaml

# Cross-domain evaluation: SBU-trained model on UCF
python tools/test.py --config configs/ucf_eval.yaml \
    --checkpoint checkpoints/scl_sbu_best.pth --root data/UCF/test

# Zero-shot, frame-by-frame video shadow detection (ViSha)
python tools/infer_video.py --config configs/visha_eval.yaml \
    --checkpoint checkpoints/scl_sbu_best.pth \
    --frames data/ViSha/video1/frames --out outputs/video1
```

## Hyperparameters (paper defaults)

| Module | Parameter | Value |
|--------|-----------|:-----:|
| ICA | EMA factor `alpha` | 0.9 |
| ICA | confidence thresholds `tau_low / tau_high` | 0.20 / 0.80 |
| ICA | down-weight `lambda_ica` | 0.50 |
| SCV | tolerance `epsilon` | 1 |
| SCV | down-weight `lambda_scv` | 0.60 |
| BRLF | warm-up `E_warm` | 5 |
| BRLF | period `P` | 1 (per-epoch) |
| Train | optimizer / lr / schedule | Adam / 5e-4 / Poly(0.7) |
| Train | epochs / batch size / size | 50 / 4 / 416×416 |

## Using the SDDNet backbone

The reported results use SDDNet (Style-guided Dual-layer Disentanglement
Network, ACM MM 2023) as the backbone. This repo ships a small `ReferenceUNet`
so the pipeline runs out of the box; to reproduce the paper, wrap the official
SDDNet via `scl.backbone.SDDNetAdapter` and map its outputs to `BackboneOutput`:

```python
logits  <- SDDNet final shadow logits (pre-sigmoid)
l_rec   <- SDDNet reconstruction-consistency loss
l_aux   <- SDDNet style-auxiliary loss
```

Official SDDNet: <https://github.com/rmcong/SDDNet_ACMMM23>.

## Results (BER ↓)

| Method | Venue | ISTD | SBU | UCF |
|--------|-------|:----:|:---:|:---:|
| SDDNet (baseline) | MM'23 | 1.27 | 2.94 | 6.59 |
| SILT† | ICCV'23 | 1.16 | 4.19† | 7.23† |
| AdapterShadow | ESWA'25 | **0.86** | 2.75 | <u>6.35</u> |
| **SCL-Shadow (ours)** | — | <u>1.14</u> | **2.70** | **6.20** |

**Bold**: best. <u>Underline</u>: second best.
†SILT evaluates SBU/UCF on its re-annotated SBU-Refine test set and is not
directly comparable on those two columns.

Relative to the SDDNet baseline, SCL-Shadow reduces BER by **8.16% on SBU** and
**10.24% on ISTD**, and achieves state-of-the-art BER on **SBU** and **UCF**.

### Ablation (BER ↓)

| ICA | SCV | BRLF | SBU | ISTD |
|:---:|:---:|:---:|:---:|:----:|
|     |     |      | 2.94 | 1.27 |
| ✓   |     |      | 2.84 | 1.23 |
|     | ✓   |      | 2.82 | 1.22 |
| ✓   | ✓   |      | 2.74 | 1.18 |
| ✓   | ✓   | ✓    | **2.70** | **1.14** |

## Citation

```bibtex
@article{xie2026sclshadow,
  title   = {Self-Calibrated Shadow Detection with Spatial Consistency Constraints under Noisy Labels},
  author  = {Xie, Jiaxuan and Chen, Xiao-Diao and Mo, Yuchang and Wu, Wen},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Submitted}
}
```

## License

Released under the [MIT License](LICENSE).
