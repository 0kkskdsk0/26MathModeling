"""Reusable Gaussian scenario weights and the exact multivariate energy score.

No date selection, outcome-based tuning or sampling occurs inside this module.
Fit FrozenStandardizer once on pre-validation context rows and reuse it.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrozenStandardizer:
    """Serializable reference-only population mean/SD, never updated at query time."""
    columns: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, reference: pd.DataFrame, fixed_scale_columns=()) -> "FrozenStandardizer":
        """Fit on caller-supplied past rows; calendar circles keep their natural scale."""
        values = reference.to_numpy(dtype=float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("At least two complete finite reference rows required")
        mean, scale = values.mean(axis=0), values.std(axis=0, ddof=0)
        for i, column in enumerate(reference.columns):
            if column in fixed_scale_columns:
                mean[i], scale[i] = 0.0, 1.0
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(tuple(reference.columns), tuple(mean), tuple(scale))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply stored parameters; incomplete raw contexts are rejected."""
        values = frame.loc[:, list(self.columns)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Incomplete context: missing causal lag observations")
        return pd.DataFrame((values - self.mean) / self.scale,
                            index=frame.index, columns=self.columns)

    def to_dict(self) -> dict:
        """Return JSON-safe parameters for direct downstream reuse."""
        return {"columns": list(self.columns), "mean": list(self.mean),
                "scale": list(self.scale)}

    @classmethod
    def from_dict(cls, data: dict) -> "FrozenStandardizer":
        """Restore a frozen scaler without fitting to new observations."""
        return cls(tuple(data["columns"]), tuple(data["mean"]), tuple(data["scale"]))


def gaussian_weights(history_features, target_features, h: float) -> np.ndarray:
    """Return normalized weights for one or many targets against the same pool.

    history_features: (K, p); target_features: (p,) or (D, p).
    Result: (K,) or (D, K). Distance is Euclidean in already scaled feature
    coordinates. h=inf explicitly denotes the uniform limiting distribution.
    Subtracting the smallest squared distance prevents exponential underflow.
    """
    history = np.asarray(history_features, dtype=float)
    target = np.asarray(target_features, dtype=float)
    single = target.ndim == 1
    if history.ndim != 2 or target.ndim not in (1, 2) or not len(history):
        raise ValueError("Expected nonempty (K,p) history and (p,) or (D,p) target")
    target = np.atleast_2d(target)
    if history.shape[1] != target.shape[1]:
        raise ValueError("Feature dimensions must match")
    if not np.isfinite(history).all() or not np.isfinite(target).all():
        raise ValueError("Features must be finite")
    if np.isnan(h) or h <= 0:
        raise ValueError("Bandwidth must be positive (or positive infinity)")
    if np.isinf(h):
        weights = np.full((len(target), len(history)), 1 / len(history))
    else:
        squared = ((target[:, None, :] - history[None, :, :]) ** 2).sum(axis=2)
        shifted = squared - squared.min(axis=1, keepdims=True)
        # Dividing sequentially also handles extremely small positive bandwidths.
        with np.errstate(over="ignore"):
            logits = -0.5 * (shifted / h) / h
        unnormalized = np.exp(logits)
        weights = unnormalized / unnormalized.sum(axis=1, keepdims=True)
    return weights[0] if single else weights


def effective_scenarios(weights) -> np.ndarray:
    """Compute K_eff=1/sum(pi**2), a concentration diagnostic, not a score."""
    weights = np.asarray(weights, dtype=float)
    return 1 / np.sum(weights ** 2, axis=-1)


def pairwise_distances(x, y=None) -> np.ndarray:
    """Exact Euclidean distances; no coordinate normalization or mean prediction."""
    x = np.asarray(x, dtype=float)
    y = x if y is None else np.asarray(y, dtype=float)
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)


def energy_score(scenarios, truth, weights) -> float:
    """Full discrete ES, including its negative half pairwise-distance term."""
    x, y, w = map(lambda a: np.asarray(a, dtype=float), (scenarios, truth, weights))
    if x.ndim != 2 or y.shape != (x.shape[1],) or w.shape != (len(x),):
        raise ValueError("Expected (K,T) scenarios, (T,) truth and (K,) weights")
    if not all(np.isfinite(v).all() for v in (x, y, w)):
        raise ValueError("ES inputs must be finite")
    if (w < 0).any() or not np.isclose(w.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("ES requires normalized nonnegative probabilities")
    return float(w @ np.linalg.norm(x - y, axis=1) - 0.5 * w @ pairwise_distances(x) @ w)


def weights_for_day(feature_frame: pd.DataFrame, target_day, history_days,
                    scaler: FrozenStandardizer, h: float) -> pd.Series:
    """Date-safe downstream API; use the caller's explicit frozen candidate pool."""
    day, dates = pd.Timestamp(target_day), pd.DatetimeIndex(history_days)
    if dates.empty or dates.has_duplicates or not (dates < day).all():
        raise ValueError("Every unique scenario date must precede the target day")
    historical = scaler.transform(feature_frame.loc[dates])
    target = scaler.transform(feature_frame.loc[[day]])
    return pd.Series(gaussian_weights(historical, target.iloc[0], h),
                     index=dates, name="weight")
