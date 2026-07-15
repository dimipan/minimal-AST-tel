"""Latent-absorption view: the novelty x yield quadrant plane.

This is a second, finer-grained phase view that sits *inside* phase_space.py. Where
phase_space collapses recurrence into one SI axis, this module decomposes the two
primitives SI is built from -- orthogonal novelty and task yield -- and reads off
four quadrants. It consumes the same ``ast-tau-v1.1`` contract and imports no
binding, no monitor.

Both axes are already in every trajectory record:

    ON  orthogonal novelty   mean of the per-dimension `eta` the monitor exports.
                             High = this exchange's evidence expanded the latent
                             subspace; low = it fell inside ground already covered.
    Y   task yield           gain this exchange produced, normalised to [0, 1]
                             against the trajectory's own peak gain.

The four quadrants (the scheme is from the latent-absorption benchmark; the
telemetry driving it is AST's):

                        high yield              low yield
    high novelty    PRODUCTIVE EXPANSION    UNPRODUCTIVE EXPANSION
    low novelty     USEFUL RECURRENCE       LATENT ABSORPTION

    PRODUCTIVE EXPANSION    new ground, and it pays -- healthy progress.
    USEFUL RECURRENCE       revisiting old ground, and it still pays -- e.g. a
                            verification step that resolves something.
    UNPRODUCTIVE EXPANSION  new ground that yields nothing -- wandering, the
                            geometric signature of drift / a distractor.
    LATENT ABSORPTION       old ground that yields nothing -- the model is stuck
                            in a region it has already exhausted. THIS is the
                            low-yield recurrence AST's SI is designed to catch;
                            it is the same corner phase_space.py calls "churn".

Why this is worth having on top of the SI axis: SI answers "is this a stall?"
The quadrants answer "what KIND of non-progress is this?" -- absorption (stuck in
place) is a different failure from unproductive expansion (drifting into novel but
worthless territory), and an intervention aimed at one is wrong for the other.

The membership scores follow the benchmark's soft product form, so each is a
continuous score in [0, 1] rather than a hard label:

    PE = ON * Y
    UR = (1 - ON) * Y
    UE = ON * (1 - Y)
    LA = (1 - ON) * (1 - Y)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from substrate_binding import load_trajectory

PRODUCTIVE_EXPANSION = "productive_expansion"
USEFUL_RECURRENCE = "useful_recurrence"
UNPRODUCTIVE_EXPANSION = "unproductive_expansion"
LATENT_ABSORPTION = "latent_absorption"

QUADRANTS = (
    PRODUCTIVE_EXPANSION,
    USEFUL_RECURRENCE,
    UNPRODUCTIVE_EXPANSION,
    LATENT_ABSORPTION,
)
QUADRANT_COLOURS = {
    PRODUCTIVE_EXPANSION: "#1f77b4",
    USEFUL_RECURRENCE: "#2ca02c",
    UNPRODUCTIVE_EXPANSION: "#ff7f0e",
    LATENT_ABSORPTION: "#d62728",
}


def _eta(record: Mapping[str, Any]) -> Mapping[str, float]:
    return record.get("eta", {})


def _gain(record: Mapping[str, Any]) -> float:
    if "gain_total" in record:
        return float(record["gain_total"])
    if "gains" in record:
        return float(sum(record["gains"].values()))
    return 0.0


def orthogonal_novelty(record: Mapping[str, Any]) -> float:
    """ON: mean per-dimension orthogonal novelty for this exchange.

    The monitor exports `eta` only for dimensions that contributed to SI. An
    exchange with no contributing dimensions (the first exchange, or a pure
    gain-only exchange with no prior subspace) has, by convention, maximal
    novelty: there is no established subspace for it to fall inside.
    """
    eta = _eta(record)
    if not eta:
        return 1.0
    return float(np.mean(list(eta.values())))


YIELD_FLOOR = 0.05   # gain at/above this is unambiguously "paying"; matches the
                     # yield cut phase_space.py uses, so the two views agree on
                     # what counts as zero yield.


def normalised_yield(
    records: Sequence[Mapping[str, Any]], *, floor: float = YIELD_FLOOR
) -> np.ndarray:
    """Y in [0, 1]: did this exchange gain MEANINGFULLY?

    A soft ramp against a small ABSOLUTE floor, not a peak-relative scale. The
    boundary is "did anything of substance resolve here", so a healthy run whose
    exchanges each gain a little must read as high-yield throughout -- which a
    peak-relative scale gets wrong, demoting every modest gain below the single
    largest one. Gains at or above `floor` saturate to 1; below it they ramp to 0.

    A run that never gains has all-zero yield, correctly: every exchange is then
    recurrence or expansion with no payoff, and the novelty axis decides which.
    """
    gains = np.array([_gain(r) for r in records], dtype=np.float64)
    return np.clip(gains / floor, 0.0, 1.0)


@dataclass(frozen=True)
class AbsorptionPoint:
    t: int
    on: float                 # orthogonal novelty
    y: float                  # normalised yield
    scores: Mapping[str, float]   # soft membership in each quadrant
    quadrant: str             # hard assignment = argmax of scores
    stall_gt: bool


def quadrant_scores(on: float, y: float) -> dict[str, float]:
    on = float(np.clip(on, 0.0, 1.0))
    y = float(np.clip(y, 0.0, 1.0))
    return {
        PRODUCTIVE_EXPANSION: on * y,
        USEFUL_RECURRENCE: (1.0 - on) * y,
        UNPRODUCTIVE_EXPANSION: on * (1.0 - y),
        LATENT_ABSORPTION: (1.0 - on) * (1.0 - y),
    }


def trajectory_to_absorption(
    exchanges: Sequence[Mapping[str, Any]],
) -> list[AbsorptionPoint]:
    ys = normalised_yield(exchanges)
    points = []
    for record, y in zip(exchanges, ys):
        on = orthogonal_novelty(record)
        scores = quadrant_scores(on, y)
        quadrant = max(scores, key=scores.get)
        gt = record.get("gt", record.get("ground_truth", {}))
        points.append(AbsorptionPoint(
            t=int(record["t"]),
            on=on,
            y=float(y),
            scores=scores,
            quadrant=quadrant,
            stall_gt=bool(gt.get("stall", False)),
        ))
    return points


def absorption_signature(points: Sequence[AbsorptionPoint]) -> dict[str, Any]:
    """Reduce a trajectory to quadrant occupancy and an absorption score.

    The headline is `latent_absorption_fraction`: the share of exchanges that fall
    in the low-novelty / low-yield corner. It is scale-free (a count over labels),
    so it transfers across substrates -- the same reason phase_space's churn
    fraction does.
    """
    if not points:
        return {"n": 0}
    quadrants = [p.quadrant for p in points]
    mean_scores = {
        q: float(np.mean([p.scores[q] for p in points])) for q in QUADRANTS
    }
    return {
        "n": len(points),
        "latent_absorption_fraction": float(
            np.mean([q == LATENT_ABSORPTION for q in quadrants])
        ),
        "productive_expansion_fraction": float(
            np.mean([q == PRODUCTIVE_EXPANSION for q in quadrants])
        ),
        "useful_recurrence_fraction": float(
            np.mean([q == USEFUL_RECURRENCE for q in quadrants])
        ),
        "unproductive_expansion_fraction": float(
            np.mean([q == UNPRODUCTIVE_EXPANSION for q in quadrants])
        ),
        "mean_scores": mean_scores,
        "terminal_quadrant": quadrants[-1],
        "mean_absorption_score": mean_scores[LATENT_ABSORPTION],
    }


# A trajectory is absorbed when a sustained share of its exchanges fall in the
# low-novelty / low-yield corner. Same cut and rationale as phase_space's churn
# fraction: a pure count, no per-substrate calibration.
ABSORPTION_FRACTION_CUT = 0.15


def is_absorbed(
    points: Sequence[AbsorptionPoint], *, cut: float = ABSORPTION_FRACTION_CUT
) -> bool:
    return absorption_signature(points)["latent_absorption_fraction"] >= cut


def load_absorption(path: str | Path):
    header, exchanges = load_trajectory(path)
    return header, trajectory_to_absorption(exchanges)


# ---------------------------------------------------------------------------
# Paired-decoy validation (ported idea, real telemetry)
# ---------------------------------------------------------------------------
def paired_decoy_check(
    same_geometry_high_yield: Sequence[AbsorptionPoint],
    same_geometry_low_yield: Sequence[AbsorptionPoint],
    *,
    quadrant_high: str,
    quadrant_low: str,
) -> dict[str, Any]:
    """The benchmark's key control, driven by real trajectories.

    Two runs with SIMILAR novelty geometry but DIFFERENT yield must land in
    DIFFERENT quadrants. If they don't, the classifier is ignoring the yield axis
    and reading geometry alone (or vice versa). This is what proves the plane uses
    both axes rather than one.

    Returns the observed dominant quadrant for each arm and whether the pair
    dissociates as intended.
    """
    def dominant(points):
        sig = absorption_signature(points)
        fractions = {
            q: sig[f"{q}_fraction"] for q in QUADRANTS
        }
        return max(fractions, key=fractions.get)

    high = dominant(same_geometry_high_yield)
    low = dominant(same_geometry_low_yield)
    return {
        "high_yield_quadrant": high,
        "low_yield_quadrant": low,
        "expected_high": quadrant_high,
        "expected_low": quadrant_low,
        "dissociates": bool(high != low),
        "matches_expected": bool(high == quadrant_high and low == quadrant_low),
        "mean_on_high": float(np.mean([p.on for p in same_geometry_high_yield])),
        "mean_on_low": float(np.mean([p.on for p in same_geometry_low_yield])),
        "mean_y_high": float(np.mean([p.y for p in same_geometry_high_yield])),
        "mean_y_low": float(np.mean([p.y for p in same_geometry_low_yield])),
    }
