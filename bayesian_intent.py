"""Recursive Bayesian intent estimator (BOIR).

The mathematics is preserved verbatim from the original research implementation
(Panagopoulos et al., SMC 2021). The implementation is not: the original rounded
the prior, likelihood, transition and posterior to two decimal places on every
tick, which quantises the very quantity this substrate exists to observe, and
guarded the recursion with an exact float equality test on the posterior sum.
Both are removed. Nothing else about the estimator changes.

Model
-----
Hidden state: one of N candidate goals.
Observations at tick t, per goal i: an angle a_i(t) and a path length l_i(t).

    Likelihood      L_i(t) = prod_k exp( -z_k,i(t) / (norm_k * w_k) )
    Transition      C_ii = 1 - Delta,  C_ij = Delta / (N - 1)   for i != j
    Posterior       p(t) proportional to L(t) * (C @ p(t-1))
    MAP             argmax_i p_i(t)

Both observation sources are "lower is better": a small angle and a short path to
goal i are evidence FOR goal i, so the likelihood decays exponentially in each.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_ANGLE_SCALE = 180.0
DEFAULT_PATH_SCALE = 25.0
DEFAULT_ANGLE_WEIGHT = 0.6
DEFAULT_PATH_WEIGHT = 0.4
DEFAULT_DELTA = 0.2


@dataclass
class RecursiveBayesianIntentEstimator:
    """Recursive Bayesian estimator over N goal hypotheses.

    Unlike the original class, `update` returns the full posterior rather than
    only the MAP index: AST observes the posterior trajectory, not the decision.
    """

    n_goals: int
    angle_scale: float = DEFAULT_ANGLE_SCALE
    path_scale: float = DEFAULT_PATH_SCALE
    angle_weight: float = DEFAULT_ANGLE_WEIGHT
    path_weight: float = DEFAULT_PATH_WEIGHT
    delta: float = DEFAULT_DELTA

    _posterior: np.ndarray = field(init=False)
    _transition: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.n_goals < 2:
            raise ValueError("n_goals must be at least 2")
        if not 0.0 <= self.delta < 1.0:
            raise ValueError("delta must lie in [0, 1)")
        for name in ("angle_scale", "path_scale", "angle_weight", "path_weight"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

        transition = np.full(
            (self.n_goals, self.n_goals), self.delta / (self.n_goals - 1), dtype=np.float64
        )
        np.fill_diagonal(transition, 1.0 - self.delta)
        self._transition = transition
        self.reset()

    # -- state ---------------------------------------------------------------
    def reset(self) -> None:
        """Uniform prior."""
        self._posterior = np.full(self.n_goals, 1.0 / self.n_goals, dtype=np.float64)

    @property
    def posterior(self) -> np.ndarray:
        return self._posterior.copy()

    @property
    def map_index(self) -> int:
        return int(np.argmax(self._posterior))

    @property
    def entropy(self) -> float:
        """Shannon entropy of the posterior, in nats."""
        p = np.clip(self._posterior, 1e-300, 1.0)
        return float(-np.sum(p * np.log(p)))

    # -- one tick ------------------------------------------------------------
    def likelihood(self, angles: np.ndarray, paths: np.ndarray) -> np.ndarray:
        """Observation model. Vectorised over goals; no rounding."""
        angles = np.asarray(angles, dtype=np.float64).reshape(-1)
        paths = np.asarray(paths, dtype=np.float64).reshape(-1)
        if angles.size != self.n_goals or paths.size != self.n_goals:
            raise ValueError(
                f"expected {self.n_goals} angles and {self.n_goals} paths, "
                f"got {angles.size} and {paths.size}"
            )
        if np.any(angles < 0.0) or np.any(paths < 0.0):
            raise ValueError("angles and path lengths must be non-negative")

        exponent = (
            -(angles / self.angle_scale) / self.angle_weight
            - (paths / self.path_scale) / self.path_weight
        )
        # Stabilised: the shift cancels in the posterior normalisation.
        return np.exp(exponent - exponent.max())

    def update(self, angles: np.ndarray, paths: np.ndarray) -> np.ndarray:
        """One recursive update. Returns the new posterior."""
        predicted = self._transition @ self._posterior
        unnormalised = self.likelihood(angles, paths) * predicted
        total = float(unnormalised.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise FloatingPointError(
                "posterior collapsed to zero mass; check observation scales"
            )
        self._posterior = unnormalised / total
        return self.posterior

    def run(self, angles: np.ndarray, paths: np.ndarray) -> np.ndarray:
        """Replay a whole (T, N) observation stream. Returns the (T, N) posterior
        trajectory. The estimator is NOT reset between calls: carry-over across an
        intent change is the phenomenon under diagnosis, not a bug to be removed.
        """
        angles = np.asarray(angles, dtype=np.float64)
        paths = np.asarray(paths, dtype=np.float64)
        if angles.shape != paths.shape or angles.ndim != 2:
            raise ValueError("angles and paths must both be (T, N) with equal shape")
        if angles.shape[1] != self.n_goals:
            raise ValueError(f"observation streams declare {angles.shape[1]} goals, "
                             f"estimator has {self.n_goals}")
        return np.stack([self.update(a, p) for a, p in zip(angles, paths)])


def binary_entropy(p: np.ndarray | float) -> np.ndarray | float:
    """H_b(p) in bits. Used by the decisiveness scorer."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats. Per-tick innovation of the posterior."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-300, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-300, 1.0)
    return float(np.sum(p * np.log(p / q)))
