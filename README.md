# SCL-Shadow

Official code for **"Self-Calibrated Shadow Detection with Spatial Consistency Constraints under Noisy Labels"** (submitted to *The Visual Computer*).

A training-time, backbone-agnostic framework that makes shadow detectors robust to label noise — **without changing the raw labels and with zero extra inference cost**. Built on SDDNet, it adds three training-only components:

- **ICA** — Iterative Confidence Aggregation: per-image EMA prediction history → down-weights temporally unreliable pixels.
- **SCV** — Spatial Consistency Verification: local boundary-structure agreement → down-weights spatially inconsistent regions.
- **BRLF** — Boundary Refined Loss Function: warm-up + per-epoch alternation between the original and reliability-weighted loss.

![Framework](assets/overview.png)

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
