# Shadow Detection with Self-Calibration and Spatial Constraint

> Jiaxuan Xie, Wen Wu, Xiao-Diao Chen
> Hangzhou Dianzi University &nbsp;·&nbsp; Zhejiang University of Water Resources and Electric Power

---

## Abstract

Deep learning-based shadow detection has made significant progress in recent years. However, annotation noise in training data causes existing methods to misidentify dark non-shadow regions and produce imprecise shadow boundaries. In this paper, we propose a robust shadow detection framework that identifies unreliable labels during training and reduces their negative effects. Specifically, we introduce an **Iterative Confidence Aggregation (ICA)** mechanism that accumulates pixel-wise prediction probabilities over time to identify unreliable pixels and adjust their supervision weights. We further design a **Spatial Consistency Verification (SCV)** mechanism that downweights supervision in regions with inconsistent local boundary structures. Finally, we adopt an **Alternating Loss Function (ALF)** strategy that switches between the original and weighted losses during training to improve robustness and reduce overfitting. Compared with the state-of-the-art baseline, our method reduces BER by **8.16%** on SBU and **10.20%** on ISTD.

---

## Framework

![Framework Overview](assets/overview.png)

The proposed framework consists of three complementary components built upon SDDNet:

- **ICA** — Iterative Confidence Aggregation: accumulates historical predictions via EMA to identify unreliable pixels and downweight their supervision.
- **SCV** — Spatial Consistency Verification: evaluates local boundary structure consistency to suppress supervision in spatially inconsistent regions.
- **ALF** — Alternating Loss Function: periodically switches between the original SDDNet loss and the weighted loss to balance disentanglement stability and reliability-weighted supervision.

### ICA Mechanism

![ICA Mechanism](assets/ica.png)

For each pixel, historical predictions are aggregated via EMA into a confidence score, which determines whether the pixel belongs to the high-confidence region or uncertain region. Unreliable pixels are downweighted during training.

### SCV Mechanism

![SCV Mechanism](assets/scv.png)

For each local 4x4 region, boundary-containing 2x2 blocks are counted in both the ground truth and the binarized prediction. When significant disagreement is detected, the supervision weight is reduced.

---

## Results

### Quantitative Comparison (BER ↓)

| Method | Venue | ISTD | SBU | UCF |
|--------|-------|:----:|:---:|:---:|
| SDDNet | MM'23 | 1.27 | 2.94 | 5.29 |
| SILT | ICCV'23 | 1.16 | 4.19† | 7.23† |
| AdapterShadow | ESWA'25 | **0.86** | 2.75 | 6.60 |
| **Ours** | — | <u>1.14</u> | **2.70** | **6.20** |

Bold: best. Underline: second best.
†SILT evaluates on the re-annotated SBU-Refine test set and is not directly comparable.

Our method achieves state-of-the-art BER on SBU and UCF, and competitive performance on ISTD.

---

## Code

Code will be released upon acceptance.

---

## Acknowledgment

This research was supported by the Zhejiang Provincial Natural Science Foundation of China under Grant No. LQN26F020068, and the Joint Fund of Zhejiang Provincial Natural Science Foundation of China under Grant No. LGEZ26F030002.

---

## Citation

```bibtex
@article{xie2025shadow,
  title={Shadow Detection with Self-Calibration and Spatial Constraint},
  author={Xie, Jiaxuan and Wu, Wen and Chen, Xiao-Diao},
  year={2025}
}
```
