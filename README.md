# SCL-Shadow

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20455673.svg)](https://doi.org/10.5281/zenodo.20455673)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official code for **"Self-Calibrated Shadow Detection with Spatial Consistency Constraints under Noisy Labels"** (submitted to *The Visual Computer*).

A training-time, backbone-agnostic framework that makes shadow detectors robust to label noise — **without changing the raw labels and with zero extra inference cost**. Built on SDDNet, it adds three training-only components:

- **ICA** — Iterative Confidence Aggregation: per-image EMA prediction history → down-weights temporally unreliable pixels.
- **SCV** — Spatial Consistency Verification: local boundary-structure agreement → down-weights spatially inconsistent regions.
- **BRLF** — Boundary Refined Loss Function: warm-up + per-epoch alternation between the original and reliability-weighted loss.

![Framework](assets/overview.png)

## Results (BER ↓, lower is better)

| Method | Venue | ISTD | SBU | UCF |
|--------|:-----:|:----:|:---:|:---:|
| FDRNet | ICCV'21 | 1.55 | 3.04 | 7.28 |
| SILT | ICCV'23 | 1.16 | 4.19† | 7.23† |
| SDDNet (baseline) | MM'23 | 1.27 | 2.94 | 6.59 |
| AdapterShadow | ESWA'25 | **0.86** | <u>2.75</u> | <u>6.35</u> |
| **Ours** | — | <u>1.14</u> | **2.70** | **6.20** |

**Bold**: best, <u>underline</u>: second best. †SILT evaluates SBU/UCF on its re-annotated SBU-Refine test set and is not directly comparable on those two columns.

Compared with the SDDNet baseline, our method reduces BER by **8.16%** on SBU and **10.24%** on ISTD, and reaches state-of-the-art BER on SBU and UCF.

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
