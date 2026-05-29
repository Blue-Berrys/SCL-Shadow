# Dataset Preparation

SCL-Shadow expects each dataset split as a directory of paired images and masks,
matched by filename stem:

```
<root>/
    images/   0001.jpg  0002.jpg  ...
    masks/    0001.png  0002.png  ...
```

Image and mask extensions may differ (e.g. `.jpg` images, `.png` masks). Masks
are binarised at load time (pixels > 0.5 are treated as shadow).

## Benchmarks

| Dataset | Train | Test | Notes |
|---------|:-----:|:----:|-------|
| SBU     | 4,089 | 638  | Natural scenes |
| ISTD    | 1,330 | 540  | Paired, high-quality annotations |
| UCF     |  —    | 221  | Test-only (cross-domain generalisation) |
| ViSha   |  —    | video | Zero-shot, frame-by-frame inference |

These are public benchmarks; download them from their official sources and
re-arrange into the `images/` + `masks/` layout above. Point the config files
(`configs/*.yaml`) at the resulting roots.

## Configs

- `configs/sbu.yaml`  — train on SBU, validate on the SBU test split.
- `configs/istd.yaml` — train on ISTD, validate on the ISTD test split.
- `configs/ucf_eval.yaml` — evaluate an SBU-trained model on UCF.
- `configs/visha_eval.yaml` — zero-shot video inference settings.

## Resolution

All images are resized to `416 x 416` during training and evaluation. The size
must be even because SCV operates on non-overlapping `2 x 2` blocks.
