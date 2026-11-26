"""End-to-end tests for the LocalizationUQ quick-start workflow.

These tests use randomly initialised models and synthetic CSI tensors, so they
need no dataset or pretrained checkpoints.  They check that the documented
workflow (``from_artifacts`` -> ``split_dataset`` -> ``calibrate`` ->
``predict`` / ``evaluate``) runs and returns well-formed results.
"""
import ast
import json
import re
from pathlib import Path

import lightning
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from trustworthy_uq import (AttentionDenseNet, FrozenBackboneOneSidedCQR,
                            FrozenBackboneScaleModel, LocalizationUQ)

REPO_ROOT = Path(__file__).resolve().parents[1]
SLA_LEVELS = [0.90, 0.95, 0.99]
N_SAMPLES = 40
CSI_SHAPE = (2, 64, 100)  # (real/imag, antennas, subcarriers)


def _save_checkpoint(model, path):
    """Write a Lightning-compatible checkpoint without running a Trainer."""
    torch.save({
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": lightning.__version__,
    }, path)
    return str(path)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    torch.manual_seed(0)
    out = tmp_path_factory.mktemp("artifacts")
    backbone = AttentionDenseNet().eval()
    cqr = FrozenBackboneOneSidedCQR(backbone, SLA_LEVELS)
    ascp = FrozenBackboneScaleModel(backbone)
    return {
        "adn": _save_checkpoint(backbone, out / "best_model.ckpt"),
        "cqr": _save_checkpoint(cqr, out / "best_cqr_head.ckpt"),
        "ascp": _save_checkpoint(ascp, out / "best_ascp_scale.ckpt"),
    }


@pytest.fixture(scope="module")
def dataset():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(N_SAMPLES, *CSI_SHAPE, generator=g)
    y = torch.rand(N_SAMPLES, 2, generator=g) * 1000.0  # positions in mm
    return TensorDataset(x, y)


def _make_uq(artifacts, **kwargs):
    kwargs.setdefault("device", torch.device("cpu"))
    return LocalizationUQ.from_artifacts(adn_model=artifacts["adn"], **kwargs)


# ---------------------------------------------------------------------------
# from_artifacts
# ---------------------------------------------------------------------------
def test_from_artifacts_loads_models(artifacts):
    uq = _make_uq(artifacts, cqr_model=artifacts["cqr"], ascp_model=artifacts["ascp"])
    assert isinstance(uq.backbone, AttentionDenseNet)
    assert isinstance(uq.cqr_model, FrozenBackboneOneSidedCQR)
    assert isinstance(uq.ascp_model, FrozenBackboneScaleModel)
    assert uq.sla_levels == SLA_LEVELS  # taken from the CQR head


def test_from_artifacts_accepts_subset_of_sla_levels(artifacts):
    uq = _make_uq(artifacts, cqr_model=artifacts["cqr"], sla_levels=[0.95])
    assert uq.sla_levels == [0.95]
    assert list(uq.cqr_model.sla_levels) == SLA_LEVELS


def test_from_artifacts_rejects_untrained_sla_level(artifacts):
    with pytest.raises(ValueError, match="0.8"):
        _make_uq(artifacts, cqr_model=artifacts["cqr"], sla_levels=[0.8])


# ---------------------------------------------------------------------------
# CQR: calibrate, predict, evaluate
# ---------------------------------------------------------------------------
def test_cqr_calibrate_predict_evaluate(artifacts, dataset, tmp_path):
    uq = _make_uq(artifacts, cqr_model=artifacts["cqr"])
    head, cal, test = uq.split_dataset(dataset)
    assert len(head) + len(cal) + len(test) == N_SAMPLES

    calib = uq.calibrate_cqr(cal, sla_levels=SLA_LEVELS, save_path=tmp_path / "cqr.json")
    assert set(calib["qhat_mm"]) == {str(level) for level in SLA_LEVELS}
    assert json.loads((tmp_path / "cqr.json").read_text())["qhat_mm"] == calib["qhat_mm"]

    x_test = torch.stack([test[i][0] for i in range(len(test))])
    y_test = torch.stack([test[i][1] for i in range(len(test))]).numpy()
    result = uq.predict(x_test, sla_levels=SLA_LEVELS, y_true=y_test)
    assert result["point_prediction"].shape == (len(test), 2)
    assert result["radius_mm"].shape == (len(test), len(SLA_LEVELS))
    assert np.all(np.isfinite(result["radius_mm"]))
    assert result["error_mm"].shape == (len(test),)
    assert set(result["covered"]) == {str(level) for level in SLA_LEVELS}

    metrics, preds = uq.evaluate(test, sla_levels=SLA_LEVELS)
    assert list(metrics["target_assurance"]) == SLA_LEVELS
    assert len(preds) == len(test) * len(SLA_LEVELS)
    assert metrics["true_coverage"].between(0, 1).all()


def test_cqr_predict_single_sample(artifacts, dataset):
    uq = _make_uq(artifacts, cqr_model=artifacts["cqr"])
    _, cal, _ = uq.split_dataset(dataset)
    uq.calibrate_cqr(cal, sla_levels=0.95)
    x, _ = dataset[0]
    result = uq.predict(x, sla_levels=0.95)  # a single (2, 64, 100) sample
    assert result["point_prediction"].shape == (1, 2)
    assert result["radius_mm"].shape == (1, 1)


def test_cqr_conformal_coverage_on_calibration_set(artifacts, dataset):
    """Split conformal guarantees >= target coverage on the calibration set itself."""
    uq = _make_uq(artifacts, cqr_model=artifacts["cqr"])
    _, cal, _ = uq.split_dataset(dataset)
    uq.calibrate_cqr(cal, sla_levels=SLA_LEVELS)
    metrics, _ = uq.evaluate(cal, sla_levels=SLA_LEVELS)
    for _, row in metrics.iterrows():
        assert row["true_coverage"] >= row["target_assurance"]


def test_saved_calibration_reused_via_from_artifacts(artifacts, dataset, tmp_path):
    uq = _make_uq(artifacts, cqr_model=artifacts["cqr"])
    _, cal, test = uq.split_dataset(dataset)
    uq.calibrate_cqr(cal, save_path=tmp_path / "cqr.json")
    x_test = torch.stack([test[i][0] for i in range(len(test))])
    expected = uq.predict(x_test)["radius_mm"]

    reloaded = _make_uq(artifacts, cqr_model=artifacts["cqr"],
                        calibration=tmp_path / "cqr.json")
    np.testing.assert_allclose(reloaded.predict(x_test)["radius_mm"], expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Adaptive SCP: calibrate, predict, evaluate
# ---------------------------------------------------------------------------
def test_ascp_calibrate_predict_evaluate(artifacts, dataset):
    uq = _make_uq(artifacts, ascp_model=artifacts["ascp"], sla_levels=SLA_LEVELS)
    _, cal, test = uq.split_dataset(dataset)
    calib = uq.calibrate_adaptive_scp(cal, sla_levels=SLA_LEVELS)
    assert set(calib["qhat"]) == {str(level) for level in SLA_LEVELS}

    x_test = torch.stack([test[i][0] for i in range(len(test))])
    result = uq.predict(x_test, sla_levels=SLA_LEVELS, method="ASCP")
    assert result["radius_mm"].shape == (len(test), len(SLA_LEVELS))
    assert np.all(result["radius_mm"] > 0)
    # Larger SLA level -> larger quantile of the same scores -> radius not smaller.
    assert np.all(np.diff(result["radius_mm"], axis=1) >= 0)

    metrics, preds = uq.evaluate(test, sla_levels=SLA_LEVELS, method="ASCP")
    assert list(metrics["target_assurance"]) == SLA_LEVELS
    assert len(preds) == len(test) * len(SLA_LEVELS)


# ---------------------------------------------------------------------------
# Documentation drift: the README and example must only call real methods
# ---------------------------------------------------------------------------
def _called_attributes(source, receivers):
    tree = ast.parse(source)
    return {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id in receivers}


@pytest.mark.parametrize("source", [
    pytest.param((REPO_ROOT / "examples" / "basic_workflow.py").read_text(), id="example"),
    pytest.param("\n".join(re.findall(r"```python\n(.*?)```",
                                      (REPO_ROOT / "README.md").read_text(), re.S)),
                 id="readme"),
])
def test_docs_only_use_existing_api(source):
    attrs = _called_attributes(source, {"uq", "LocalizationUQ"})
    assert attrs, "expected the snippet to use LocalizationUQ"
    missing = sorted(a for a in attrs if not hasattr(LocalizationUQ, a))
    assert not missing, f"documented methods missing from LocalizationUQ: {missing}"
