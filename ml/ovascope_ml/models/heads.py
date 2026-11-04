"""Linear heads stored as plain NumPy arrays (no pickle).

Bundles are loaded with `np.load(allow_pickle=False)`, so a tampered checkpoint
cannot execute code on load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


@dataclass
class LinearHead:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray  # (K, D) softmax, or (1, D) sigmoid
    intercept: np.ndarray  # (K,) or (1,)
    classes: list[str]
    kind: Literal["softmax", "sigmoid"] = "softmax"
    temperature: float = 1.0
    meta: dict[str, str] = field(default_factory=dict)

    def logits(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return z @ self.coef.T + self.intercept

    def scores(self, x: np.ndarray, temperature_scaled: bool = True) -> np.ndarray:
        t = self.temperature if temperature_scaled else 1.0
        lg = self.logits(x) / t
        if self.kind == "sigmoid":
            return 1.0 / (1.0 + np.exp(-lg[:, 0]))
        return softmax(lg, axis=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            scale=self.scale,
            coef=self.coef,
            intercept=self.intercept,
            classes=np.array(self.classes, dtype="U64"),
            kind=np.array(self.kind, dtype="U16"),
            temperature=np.array(self.temperature, dtype=np.float64),
            meta_keys=np.array(list(self.meta.keys()), dtype="U64"),
            meta_values=np.array(list(self.meta.values()), dtype="U512"),
        )

    @classmethod
    def load(cls, path: Path) -> LinearHead:
        with np.load(path, allow_pickle=False) as z:
            return cls(
                mean=z["mean"],
                scale=z["scale"],
                coef=z["coef"],
                intercept=z["intercept"],
                classes=[str(c) for c in z["classes"]],
                kind=str(z["kind"]),  # type: ignore[arg-type]
                temperature=float(z["temperature"]),
                meta=dict(zip((str(k) for k in z["meta_keys"]), (str(v) for v in z["meta_values"]), strict=True)),
            )

    @classmethod
    def from_sklearn(
        cls, scaler, lr, classes: list[str], kind: Literal["softmax", "sigmoid"] = "softmax"
    ) -> LinearHead:
        scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
        return cls(
            mean=scaler.mean_.astype(np.float64),
            scale=scale.astype(np.float64),
            coef=lr.coef_.astype(np.float64),
            intercept=lr.intercept_.astype(np.float64),
            classes=classes,
            kind=kind,
        )


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature minimising NLL of held-out (out-of-fold) logits. Argmax-preserving."""

    def nll(log_t: float) -> float:
        lp = log_softmax(logits / np.exp(log_t), axis=1)
        return float(-lp[np.arange(len(labels)), labels].mean())

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))
