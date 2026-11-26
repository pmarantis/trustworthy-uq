"""Online, per-sample uncertainty-aware localization.

This module turns a calibrated :class:`~trustworthy_uq.LocalizationUQ`
instance into a small, deployable engine:

* :class:`TrustedLocalizer` answers one CSI measurement at a time with a
  position estimate and a conformal *trust radius* for each SLA level.
* :meth:`TrustedLocalizer.save` writes everything needed for inference
  (backbone weights, UQ head weights, conformal offsets, SLA levels) into one
  directory, and :meth:`TrustedLocalizer.load` restores it in a new process
  without the calibration dataset.

Typical use::

    uq = LocalizationUQ(sla_levels=[0.90, 0.95, 0.99])
    uq.run_experiment(data_dir, method="CQR", adn_checkpoint=..., cqr_checkpoint=...)
    TrustedLocalizer(uq, method="CQR").save("engine/")

    # later, e.g. in a serving process
    localizer = TrustedLocalizer.load("engine/")
    result = localizer.locate(csi)          # complex (64, 100) array
    result.position_mm, result.radius(0.95), result.meets(50.0, sla=0.95)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import lightning
import numpy as np
import torch

from .localization.pipeline import LocalizationUQ

ENGINE_FORMAT = "trustworthy-uq-engine"
ENGINE_FORMAT_VERSION = 1
MANIFEST_NAME = "engine.json"
ADN_FILE = "adn_backbone.ckpt"
HEAD_FILES = {"CQR": "cqr_head.ckpt", "aSCP": "ascp_scale.ckpt"}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def canonical_method(method: str) -> str:
    """Map the accepted method spellings to ``"CQR"`` or ``"aSCP"``."""
    key = str(method).upper().replace("-", "").replace("_", "")
    if key == "CQR":
        return "CQR"
    if key in ("ASCP", "ADAPTIVESCP"):
        return "aSCP"
    raise ValueError(f"Unsupported method {method!r}. Use 'CQR' or 'aSCP'.")


def _as_array(csi) -> np.ndarray:
    if isinstance(csi, torch.Tensor):
        return csi.detach().cpu().resolve_conj().numpy()
    return np.asarray(csi)


def _is_single_sample(csi) -> bool:
    """True for one un-batched sample: complex 2-D or real 3-D."""
    arr = _as_array(csi)
    return arr.ndim == (2 if np.iscomplexobj(arr) else 3)


def csi_to_input(csi, num_antennas: int = 64, num_subcarriers: int = 100) -> torch.Tensor:
    """Convert CSI into the ``(N, 2, antennas, subcarriers)`` float tensor the ADN expects.

    Accepted inputs:

    * complex ``(antennas, subcarriers)`` or ``(N, antennas, subcarriers)``,
      i.e. the raw measurement as stored in the Nomadic ``.npy`` files;
    * real ``(2, antennas, subcarriers)`` or ``(N, 2, antennas, subcarriers)``,
      i.e. the real/imaginary stacking produced by the dataset loader.
    """
    arr = _as_array(csi)
    if np.iscomplexobj(arr):
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim != 3:
            raise ValueError(
                "Complex CSI must have shape (antennas, subcarriers) or "
                f"(N, antennas, subcarriers); got {arr.shape}.")
        arr = np.stack([arr.real, arr.imag], axis=1)
    else:
        if arr.ndim == 3:
            arr = arr[None]
        if arr.ndim != 4 or arr.shape[1] != 2:
            raise ValueError(
                "Real CSI must have shape (2, antennas, subcarriers) or "
                f"(N, 2, antennas, subcarriers); got {arr.shape}.")

    expected = (num_antennas, num_subcarriers)
    if tuple(arr.shape[2:]) != expected:
        raise ValueError(
            f"CSI has {arr.shape[2]} antennas x {arr.shape[3]} subcarriers, but the "
            f"backbone expects {expected[0]} x {expected[1]}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("CSI contains NaN or infinite values.")
    return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32)


def _level_key(level, available: Sequence[float]) -> float:
    """Return the entry of ``available`` equal to ``level`` (within 1e-9)."""
    for candidate in available:
        if abs(float(candidate) - float(level)) < 1e-9:
            return candidate
    raise ValueError(
        f"SLA level {level} is not calibrated. Calibrated levels: {list(available)}.")


def _lightning_checkpoint(model, hyper_parameters) -> dict:
    """A minimal checkpoint that ``LightningModule.load_from_checkpoint`` accepts."""
    return {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(hyper_parameters),
        "pytorch-lightning_version": lightning.__version__,
    }


# ----------------------------------------------------------------------
# Result type
# ----------------------------------------------------------------------
@dataclass
class TrustedLocation:
    """One localization result with its conformal trust radii.

    ``radius_mm[level]`` is the radius of a disk around ``position_mm`` that
    contains the true position with probability at least ``level``
    (marginal split-conformal guarantee under exchangeability with the
    calibration data).
    """

    position_mm: np.ndarray
    radius_mm: Dict[float, float]
    method: str
    error_mm: Optional[float] = None
    covered: Optional[Dict[float, bool]] = None
    latency_ms: Optional[float] = field(default=None, compare=False)

    @property
    def sla_levels(self) -> List[float]:
        return list(self.radius_mm)

    def radius(self, sla: float) -> float:
        """Trust radius in millimetres for one SLA level."""
        return self.radius_mm[_level_key(sla, self.sla_levels)]

    def meets(self, max_radius_mm: float, sla: float) -> bool:
        """True when the ``sla`` trust radius is within ``max_radius_mm``.

        Use it to decide whether the estimate is precise enough for an
        application that needs, e.g., 95% assurance of a 50 mm error.
        """
        return self.radius(sla) <= float(max_radius_mm)

    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        out = {
            "method": self.method,
            "position_mm": [float(v) for v in self.position_mm],
            "radius_mm": {str(k): float(v) for k, v in self.radius_mm.items()},
        }
        if self.error_mm is not None:
            out["error_mm"] = float(self.error_mm)
        if self.covered is not None:
            out["covered"] = {str(k): bool(v) for k, v in self.covered.items()}
        if self.latency_ms is not None:
            out["latency_ms"] = float(self.latency_ms)
        return out


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------
class TrustedLocalizer:
    """Per-sample position + trust-radius inference on a calibrated engine.

    Parameters
    ----------
    uq:
        A :class:`LocalizationUQ` whose backbone, UQ model and conformal
        calibration are already in place (for example after
        ``run_experiment`` or ``calibrate_cqr``).
    method:
        ``"CQR"`` or ``"aSCP"``.
    sla_levels:
        Default SLA levels returned by :meth:`locate`. Defaults to every
        calibrated level.
    """

    def __init__(self, uq: LocalizationUQ, method: str = "CQR",
                 sla_levels: Optional[Sequence[float]] = None,
                 metadata: Optional[dict] = None):
        self.uq = uq
        self.method = canonical_method(method)
        self.metadata = dict(metadata or {})
        self._check_ready()
        calibrated = self.calibrated_levels
        if sla_levels is None:
            self.sla_levels = calibrated
        else:
            self.sla_levels = [_level_key(level, calibrated) for level in sla_levels]

    # -- state ---------------------------------------------------------
    def _qhats(self) -> dict:
        return self.uq.cqr_qhats if self.method == "CQR" else self.uq.ascp_qhats

    def _head(self):
        return self.uq.cqr_model if self.method == "CQR" else self.uq.ascp_model

    def _check_ready(self):
        if self.uq.backbone is None:
            raise RuntimeError("No ADN backbone loaded. Call uq.load_adn(...) or uq.train_adn(...).")
        if self._head() is None:
            name = "CQR head (uq.load_cqr / uq.train_cqr)" if self.method == "CQR" \
                else "aSCP scale model (uq.load_ascp / uq.train_adaptive_scp)"
            raise RuntimeError(f"No {name} loaded.")
        if not self._qhats():
            fn = "calibrate_cqr" if self.method == "CQR" else "calibrate_adaptive_scp"
            raise RuntimeError(
                f"The {self.method} engine is not calibrated. Call uq.{fn}(...) "
                "or uq.load_calibration(...) first.")
        if self.method == "CQR":
            missing = [k for k in self._qhats()
                       if not any(abs(float(k) - float(t)) < 1e-9 for t in self.uq.cqr_model.sla_levels)]
            if missing:
                raise RuntimeError(
                    f"Calibrated SLA levels {missing} are not outputs of the CQR head "
                    f"(head levels: {list(self.uq.cqr_model.sla_levels)}).")

    @property
    def calibrated_levels(self) -> List[float]:
        return sorted(float(k) for k in self._qhats())

    @property
    def input_shape(self):
        b = self.uq.backbone
        return (2, int(b.num_antennas), int(b.num_subcarriers))

    # -- inference -----------------------------------------------------
    def locate(self, csi, sla_levels: Optional[Sequence[float]] = None,
               y_true=None) -> Union[TrustedLocation, List[TrustedLocation]]:
        """Estimate the position and trust radius for one or more CSI samples.

        Returns one :class:`TrustedLocation` for a single sample and a list
        for a batch. ``y_true`` (``(2,)`` or ``(N, 2)``, in mm) is optional
        and only adds the error and coverage fields, e.g. for monitoring.
        """
        if isinstance(sla_levels, (int, float)):
            sla_levels = [sla_levels]
        levels = (self.sla_levels if sla_levels is None
                  else [_level_key(level, self.calibrated_levels) for level in sla_levels])
        _, n_ant, n_sub = self.input_shape
        x = csi_to_input(csi, n_ant, n_sub)
        single = _is_single_sample(csi)

        true = None
        if y_true is not None:
            true = np.asarray(y_true, dtype=np.float32)
            if true.ndim == 1:
                true = true[None, :]
            if true.shape != (x.shape[0], 2):
                raise ValueError(f"y_true must have shape (2,) or ({x.shape[0]}, 2); got {true.shape}.")

        start = time.perf_counter()
        pred = self.uq.predict(x, sla_levels=list(levels), y_true=true,
                               method="CQR" if self.method == "CQR" else "aSCP")
        latency_ms = (time.perf_counter() - start) * 1000.0 / x.shape[0]

        results = []
        for i in range(x.shape[0]):
            radii = {float(level): float(pred["radius_mm"][i, j]) for j, level in enumerate(levels)}
            error = covered = None
            if true is not None:
                error = float(pred["error_mm"][i])
                covered = {float(level): bool(error <= radii[float(level)]) for level in levels}
            results.append(TrustedLocation(
                position_mm=np.asarray(pred["point_prediction"][i], dtype=float),
                radius_mm=radii, method=self.method, error_mm=error,
                covered=covered, latency_ms=latency_ms))
        return results[0] if single else results

    # -- persistence ---------------------------------------------------
    def save(self, directory, overwrite: bool = False) -> Path:
        """Write a self-contained engine directory and return its manifest path.

        The directory holds ``engine.json`` plus Lightning-compatible
        checkpoints for the backbone and the UQ head, so the files can also
        be loaded with ``LocalizationUQ.load_adn`` / ``load_cqr`` /
        ``load_ascp`` directly.
        """
        from . import __version__

        directory = Path(directory)
        manifest_path = directory / MANIFEST_NAME
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(
                f"{manifest_path} already exists. Pass overwrite=True to replace it.")
        directory.mkdir(parents=True, exist_ok=True)

        backbone = self.uq.backbone
        adn_hparams = {
            "in_features": int(backbone.denseBlock1.conv1.in_channels),
            "num_antennas": int(backbone.num_antennas),
            "num_subcarriers": int(backbone.num_subcarriers),
        }
        torch.save(_lightning_checkpoint(backbone, adn_hparams), directory / ADN_FILE)

        head = self._head()
        head_file = HEAD_FILES[self.method]
        torch.save(_lightning_checkpoint(head, head.hparams), directory / head_file)

        calibration_info = (self.uq.cqr_calibration_info if self.method == "CQR"
                            else self.uq.ascp_calibration_info)
        manifest = {
            "format": ENGINE_FORMAT,
            "format_version": ENGINE_FORMAT_VERSION,
            "trustworthy_uq_version": __version__,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task": "2D-localization",
            "units": "mm",
            "method": self.method,
            "sla_levels": [float(level) for level in self.sla_levels],
            "qhat": {str(float(k)): float(v) for k, v in self._qhats().items()},
            "num_calibration_samples": calibration_info.get("num_calibration_samples"),
            "input_shape": list(self.input_shape),
            "files": {"adn": ADN_FILE, "head": head_file},
            "adn_hparams": adn_hparams,
            "head_hparams": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                             for k, v in dict(head.hparams).items()},
            "metadata": self.metadata,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return manifest_path

    @classmethod
    def load(cls, directory, device=None, sla_levels: Optional[Sequence[float]] = None,
             **uq_kwargs) -> "TrustedLocalizer":
        """Restore an engine written by :meth:`save`; no dataset is needed."""
        directory = Path(directory)
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"No {MANIFEST_NAME} in {directory}.")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != ENGINE_FORMAT:
            raise ValueError(f"{manifest_path} is not a trustworthy-uq engine manifest.")
        if int(manifest.get("format_version", 0)) > ENGINE_FORMAT_VERSION:
            raise ValueError(
                f"Engine format version {manifest['format_version']} is newer than this "
                f"package supports ({ENGINE_FORMAT_VERSION}). Upgrade trustworthy-uq.")

        method = canonical_method(manifest["method"])
        if device is not None:
            uq_kwargs["device"] = torch.device(device)
        uq = LocalizationUQ(sla_levels=manifest["sla_levels"], **uq_kwargs)
        uq.load_adn(str(directory / manifest["files"]["adn"]))
        qhat = {str(float(k)): float(v) for k, v in manifest["qhat"].items()}
        info = {"num_calibration_samples": manifest.get("num_calibration_samples"),
                "sla_levels": sorted(float(k) for k in qhat)}
        if method == "CQR":
            uq.load_cqr(str(directory / manifest["files"]["head"]))
            uq.cqr_qhats = qhat
            uq.cqr_calibration_info = {**info, "qhat_mm": qhat}
        else:
            uq.load_ascp(str(directory / manifest["files"]["head"]))
            uq.ascp_qhats = qhat
            uq.ascp_calibration_info = {**info, "qhat": qhat}
        uq.sla_levels = list(manifest["sla_levels"])
        return cls(uq, method=method,
                   sla_levels=sla_levels if sla_levels is not None else manifest["sla_levels"],
                   metadata=manifest.get("metadata"))

    def __repr__(self):
        return (f"TrustedLocalizer(method={self.method!r}, sla_levels={self.sla_levels}, "
                f"input_shape={self.input_shape})")
