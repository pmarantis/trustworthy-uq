"""Tests for the online engine (TrustedLocalizer) and its save/load bundle.

They use a randomly initialised ADN and synthetic CSI, so no dataset or
pretrained checkpoint is needed.
"""

import json

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from trustworthy_uq import (AttentionDenseNet, FrozenBackboneOneSidedCQR,
                            FrozenBackboneScaleModel, LocalizationUQ,
                            TrustedLocalizer, TrustedLocation)
from trustworthy_uq.data.nomadic import normalize_augment_method
from trustworthy_uq.online import MANIFEST_NAME, csi_to_input

LEVELS = [0.90, 0.95, 0.99]
N_ANT, N_SUB = 64, 100


def _synthetic(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, N_ANT, N_SUB, generator=g)
    y = torch.randn(n, 2, generator=g) * 50.0
    return x, y


def _engine(method):
    torch.manual_seed(0)
    uq = LocalizationUQ(sla_levels=LEVELS, device=torch.device("cpu"))
    uq.backbone = AttentionDenseNet().eval()
    x, y = _synthetic(40, seed=1)
    calibration = TensorDataset(x, y)
    if method == "CQR":
        uq.cqr_model = FrozenBackboneOneSidedCQR(uq.backbone, LEVELS).eval()
        uq.calibrate_cqr(calibration, LEVELS)
    else:
        uq.ascp_model = FrozenBackboneScaleModel(uq.backbone).eval()
        uq.calibrate_adaptive_scp(calibration, LEVELS)
    return uq


@pytest.fixture(scope="module")
def cqr_uq():
    return _engine("CQR")


@pytest.fixture(scope="module")
def ascp_uq():
    return _engine("aSCP")


# ----------------------------------------------------------------------
# Input conversion
# ----------------------------------------------------------------------
def test_complex_csi_is_stacked_like_the_dataset():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(N_ANT, N_SUB)) + 1j * rng.normal(size=(N_ANT, N_SUB))
    x = csi_to_input(raw)
    assert x.shape == (1, 2, N_ANT, N_SUB)
    assert x.dtype == torch.float32
    np.testing.assert_allclose(x[0, 0].numpy(), raw.real, rtol=1e-6)
    np.testing.assert_allclose(x[0, 1].numpy(), raw.imag, rtol=1e-6)


def test_batched_and_real_inputs():
    raw = np.ones((3, N_ANT, N_SUB), dtype=np.complex64)
    assert csi_to_input(raw).shape == (3, 2, N_ANT, N_SUB)
    assert csi_to_input(torch.zeros(2, N_ANT, N_SUB)).shape == (1, 2, N_ANT, N_SUB)
    assert csi_to_input(torch.zeros(5, 2, N_ANT, N_SUB)).shape == (5, 2, N_ANT, N_SUB)


@pytest.mark.parametrize("bad", [
    np.zeros((N_ANT, N_SUB)),                        # real but not stacked
    np.zeros((2, 32, N_SUB)),                        # wrong antenna count
    np.zeros((N_ANT, 50), dtype=np.complex64),       # wrong subcarrier count
])
def test_bad_csi_shapes_raise(bad):
    with pytest.raises(ValueError):
        csi_to_input(bad)


def test_non_finite_csi_raises():
    x = np.zeros((2, N_ANT, N_SUB))
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        csi_to_input(x)


# ----------------------------------------------------------------------
# Readiness checks
# ----------------------------------------------------------------------
def test_uncalibrated_engine_is_rejected():
    uq = LocalizationUQ(sla_levels=LEVELS, device=torch.device("cpu"))
    with pytest.raises(RuntimeError, match="backbone"):
        TrustedLocalizer(uq)
    uq.backbone = AttentionDenseNet().eval()
    with pytest.raises(RuntimeError, match="CQR head"):
        TrustedLocalizer(uq)
    uq.cqr_model = FrozenBackboneOneSidedCQR(uq.backbone, LEVELS).eval()
    with pytest.raises(RuntimeError, match="not calibrated"):
        TrustedLocalizer(uq)


def test_uncalibrated_sla_level_is_rejected(cqr_uq):
    localizer = TrustedLocalizer(cqr_uq)
    with pytest.raises(ValueError, match="not calibrated"):
        localizer.locate(np.zeros((2, N_ANT, N_SUB)), sla_levels=[0.8])


# ----------------------------------------------------------------------
# locate()
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ["CQR", "aSCP"])
def test_locate_matches_predict(method, cqr_uq, ascp_uq):
    uq = cqr_uq if method == "CQR" else ascp_uq
    localizer = TrustedLocalizer(uq, method=method)
    x, y = _synthetic(4, seed=7)

    expected = uq.predict(x, sla_levels=LEVELS, method=method)
    batch = localizer.locate(x, y_true=y)
    assert isinstance(batch, list) and len(batch) == 4
    for i, res in enumerate(batch):
        np.testing.assert_allclose(res.position_mm, expected["point_prediction"][i], rtol=1e-5)
        np.testing.assert_allclose([res.radius(l) for l in LEVELS], expected["radius_mm"][i], rtol=1e-5)
        err = float(np.linalg.norm(res.position_mm - y[i].numpy()))
        assert res.error_mm == pytest.approx(err, rel=1e-4)
        assert res.covered == {l: err <= res.radius(l) for l in LEVELS}

    single = localizer.locate(x[0])
    assert isinstance(single, TrustedLocation)
    np.testing.assert_allclose(single.position_mm, batch[0].position_mm, rtol=1e-5)


def test_locate_single_sla_and_requirement(cqr_uq):
    localizer = TrustedLocalizer(cqr_uq, sla_levels=[0.95])
    res = localizer.locate(np.zeros((2, N_ANT, N_SUB), dtype=np.float32))
    assert res.sla_levels == [0.95]
    r = res.radius(0.95)
    assert res.meets(r + 1.0, sla=0.95)
    assert not res.meets(r - 1.0, sla=0.95)
    d = res.to_dict()
    json.dumps(d)
    assert d["method"] == "CQR" and set(d["radius_mm"]) == {"0.95"}


# ----------------------------------------------------------------------
# save() / load()
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ["CQR", "aSCP"])
def test_save_load_round_trip(method, cqr_uq, ascp_uq, tmp_path):
    uq = cqr_uq if method == "CQR" else ascp_uq
    original = TrustedLocalizer(uq, method=method, metadata={"backbone": "random-init"})
    manifest_path = original.save(tmp_path / "engine")

    manifest = json.loads(manifest_path.read_text())
    assert manifest["method"] == method
    assert manifest["sla_levels"] == LEVELS
    assert manifest["input_shape"] == [2, N_ANT, N_SUB]
    assert manifest["num_calibration_samples"] == 40
    assert manifest["metadata"] == {"backbone": "random-init"}

    restored = TrustedLocalizer.load(tmp_path / "engine", device="cpu")
    assert restored.method == method
    assert restored.sla_levels == LEVELS
    x, _ = _synthetic(3, seed=11)
    for a, b in zip(original.locate(x), restored.locate(x)):
        np.testing.assert_allclose(a.position_mm, b.position_mm, rtol=1e-5, atol=1e-4)
        for level in LEVELS:
            assert a.radius(level) == pytest.approx(b.radius(level), rel=1e-5, abs=1e-4)


def test_saved_checkpoints_load_with_lightning(cqr_uq, tmp_path):
    TrustedLocalizer(cqr_uq).save(tmp_path)
    uq = LocalizationUQ(device=torch.device("cpu"))
    uq.load_adn(str(tmp_path / "adn_backbone.ckpt"))
    uq.load_cqr(str(tmp_path / "cqr_head.ckpt"))
    assert uq.sla_levels == LEVELS


def test_save_refuses_to_overwrite(cqr_uq, tmp_path):
    localizer = TrustedLocalizer(cqr_uq)
    localizer.save(tmp_path)
    with pytest.raises(FileExistsError):
        localizer.save(tmp_path)
    localizer.save(tmp_path, overwrite=True)


def test_load_rejects_foreign_manifest(tmp_path):
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"format": "something-else"}))
    with pytest.raises(ValueError):
        TrustedLocalizer.load(tmp_path)


# ----------------------------------------------------------------------
# Dataset augmentation names
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name, expected", [
    ("random_attenuation", "random attenuation"),
    ("random attenuation", "random attenuation"),
    ("vanilla", "vanilla"),
    ("no_augmentation", None),
    (None, None),
])
def test_augment_method_aliases(name, expected):
    assert normalize_augment_method(name) == expected


def test_unknown_augment_method_raises():
    with pytest.raises(ValueError):
        normalize_augment_method("mixup")
