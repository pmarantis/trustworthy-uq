# trustworthy-uq

A simple Python package for uncertainty-aware CSI localization experiments used in the 6G-DALI work.

## Included

- AttentionDenseNet (ADN) localization backbone
- One-sided multi-SLA CQR
- Adaptive SCP with learned scale model
- KUL/Nomadic CSI dataset loader
- Training/loading of ADN and CQR models
- Conformal calibration for one or multiple SLA levels
- Batch inference and single-sample prediction disks
- Coverage, breach rate and radius evaluation
- CSV and Plotly artifact generation
- Optional MLflow adapter

## Basic workflow

```python
import torch
from trustworthy_uq import LocalizationUQ

# Load a trained ADN backbone and CQR head (CQR head is optional; use
# ascp_model=... for Adaptive SCP, calibration=... to reuse a saved calibration).
uq = LocalizationUQ.from_artifacts(
    adn_model="best_model.ckpt",
    cqr_model="best_cqr_head.ckpt",
    sla_levels=[0.95],
)

# Build a dataset and split it into head-training / calibration / test parts.
dataset = uq.build_pooled_dataset("data/nomadic_dataset/ULA_lab_LoS")
_, calibration_dataset, test_dataset = uq.split_dataset(dataset)

uq.calibrate_cqr(calibration_dataset, sla_levels=[0.95])

# X_test: CSI samples of shape (N, 2, 64, 100), or a single (2, 64, 100) sample.
X_test = torch.stack([test_dataset[i][0] for i in range(8)])
result = uq.predict(X_test, sla_levels=[0.95])
metrics, predictions = uq.evaluate(test_dataset, sla_levels=[0.95])
```

`from_artifacts` is a shortcut for constructing `LocalizationUQ(...)` and calling
`load_adn`, `load_cqr`, `load_ascp` and `load_calibration` as needed.

The package keeps the CQR head multi-SLA: it is trained once for the configured SLA levels, while conformal calibration can be performed for one or more of those levels.

## Online inference

`TrustedLocalizer` wraps a calibrated `LocalizationUQ` for per-sample use and saves it as a self-contained engine directory (`engine.json` plus Lightning-compatible checkpoints):

```python
from trustworthy_uq import TrustedLocalizer

TrustedLocalizer(uq, method="CQR").save("engine/")   # after calibration

localizer = TrustedLocalizer.load("engine/")         # no dataset needed
result = localizer.locate(csi)    # complex (64, 100) or real (2, 64, 100); batches give a list
result.position_mm, result.radius(0.95), result.meets(25.0, sla=0.95), result.to_dict()
```
