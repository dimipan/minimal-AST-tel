"""Phase-space layer for AST trajectories.

This module reads the shared ``ast-tau-v1.1`` contract and NOTHING else. It does
not import a substrate, a binding, or the frozen monitor. Give it any trajectory
that any binding produced and it will place every exchange in a regime plane. That
independence is the whole point: the phase space is defined over telemetry, so it
is substrate-agnostic by construction, exactly as the monitor is.

The plane
---------
Each exchange carries three telemetry quantities that together name a regime:

    yield        gain_total          did this exchange resolve anything?
    friction     SI                  is evidence recurring without producing?
    pressure     PE_total            how much of the schema is still unresolved?

The regime table AST's design predicts (and which the regime portrait in
plotting.py draws) is:

    regime              yield   friction   pressure
    ---------------------------------------------------------
    PRODUCTIVE          > 0     low         any        resolving normally
    CHURN (stall)       ~ 0     high        high       the pathological corner
    CONVERGED           ~ 0     low         low        done; nothing left to do
    QUIET-INCOMPLETE    ~ 0     low         high       stalled WITHOUT recurrence
                                                       (starved, not spinning)

CHURN is the corner AST exists to find: work happening, nothing acquired, and the
schema still open. CONVERGED is its innocent twin -- also zero-yield, also
low-friction, but the schema is closed, so there is nothing to detect. Telling
those two apart needs both SI and PE, which is precisely why the monitor reports
two channels rather than one. QUIET-INCOMPLETE is the fourth cell: a process that
has simply stopped receiving informative evidence, without repeating itself. It is
not a recurrence stall, and SI is right not to flag it; the phase space surfaces it
so it is not silently lumped in with CONVERGED.

The pre-registered prediction
-----------------------------
The trajectory SHAPE separates outcomes, not any single point. A run that ends
resolved passes through PRODUCTIVE and settles into CONVERGED: friction only rises
(if at all) after pressure has drained. A run that stalls enters CHURN and stays:
friction is high WHILE pressure is still high. So the diagnostic is the order in
which the plane is traversed -- friction-after-drain is benign, friction-with-
pressure is pathological. `phase_signature` reduces a whole trajectory to that
distinction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from substrate_binding import load_trajectory

# Regime labels
PRODUCTIVE = "productive"
CHURN = "churn"
CONVERGED = "converged"
QUIET_INCOMPLETE = "quiet_incomplete"

REGIME_ORDER = (PRODUCTIVE, CHURN, QUIET_INCOMPLETE, CONVERGED)
REGIME_COLOURS = {
    PRODUCTIVE: "#1f77b4",
    CHURN: "#d62728",
    QUIET_INCOMPLETE: "#ff7f0e",
    CONVERGED: "#2ca02c",
}


@dataclass(frozen=True)
class PhaseThresholds:
    """Cut points for the plane.

    These are reading conventions for classifying telemetry that already exists,
    not monitor calibration. `si` reuses the same stall threshold the substrate
    declared, so the phase plane and the detector agree on what "high friction"
    means. `yield_` and `pressure` are read off the trajectory's own scale (see
    `from_trajectory`) so the layer is not tied to any one substrate's units.
    """

    yield_: float = 0.02
    si: float = 0.20
    pressure: float = 0.0    # set per-trajectory by from_trajectory

    @classmethod
    def from_trajectory(
        cls, exchanges: Sequence[Mapping[str, Any]], *, si_threshold: float = 0.20
    ) -> "PhaseThresholds":
        """Pressure cut = midpoint between the trajectory's PE floor and ceiling.

        PE is representation- and schema-dependent, so a fixed absolute cut would
        not transfer between substrates -- the same lesson SI's scale taught us.
        The midpoint of the observed range is scale-free and needs no tuning.
        """
        pe = np.array([_pressure(r) for r in exchanges])
        pressure_cut = float((pe.min() + pe.max()) / 2.0) if pe.size else 0.0
        return cls(yield_=0.02, si=si_threshold, pressure=pressure_cut)


def _yield(record: Mapping[str, Any]) -> float:
    if "gain_total" in record:
        return float(record["gain_total"])
    if "gains" in record:
        return float(sum(record["gains"].values()))
    return 0.0


def _friction(record: Mapping[str, Any]) -> float:
    return float(record["SI"] if "SI" in record else record["si"])


def _pressure(record: Mapping[str, Any]) -> float:
    return float(record["PE_total"] if "PE_total" in record else record["pe_total"])


def _stall(record: Mapping[str, Any]) -> bool:
    gt = record.get("gt", record.get("ground_truth", {}))
    return bool(gt.get("stall", False))


def classify_exchange(
    record: Mapping[str, Any], thresholds: PhaseThresholds
) -> str:
    """Place one exchange in the regime plane. Reads the JSONL contract, and also
    tolerates the in-memory record shape returned by run_binding."""
    yield_ = _yield(record)
    si = _friction(record)
    pe = _pressure(record)

    if yield_ > thresholds.yield_:
        return PRODUCTIVE
    # zero-yield: the friction axis splits stall from rest
    if si >= thresholds.si:
        return CHURN
    # low friction and low yield: converged if the schema is closed, else starved
    return CONVERGED if pe < thresholds.pressure else QUIET_INCOMPLETE


@dataclass(frozen=True)
class PhasePoint:
    t: int
    yield_: float
    friction: float
    pressure: float
    knowledge: float
    regime: str
    stall_gt: bool


def trajectory_to_phase(
    exchanges: Sequence[Mapping[str, Any]],
    *,
    si_threshold: float = 0.20,
    thresholds: PhaseThresholds | None = None,
) -> list[PhasePoint]:
    thresholds = thresholds or PhaseThresholds.from_trajectory(
        exchanges, si_threshold=si_threshold
    )
    points = []
    for record in exchanges:
        points.append(PhasePoint(
            t=int(record["t"]),
            yield_=_yield(record),
            friction=_friction(record),
            pressure=_pressure(record),
            knowledge=float(record["knowledge_mean"]),
            regime=classify_exchange(record, thresholds),
            stall_gt=_stall(record),
        ))
    return points


def phase_signature(points: Sequence[PhasePoint]) -> dict[str, Any]:
    """Reduce a trajectory to its pre-registered diagnostic.

    The claim is about ORDER: does friction rise before or after pressure drains?

      benign      friction stays low until pressure has fallen -> settles CONVERGED
      pathological friction is high WHILE pressure is still high -> sits in CHURN

    Reported as `churn_fraction` (share of exchanges in the high-SI/high-PE corner)
    and `friction_while_pressured` (mean SI over exchanges whose PE is still above
    the pressure cut). A high value of the latter is the pathological signature.
    """
    if not points:
        return {"n": 0}

    pressures = np.array([p.pressure for p in points])
    frictions = np.array([p.friction for p in points])
    pressure_cut = float((pressures.min() + pressures.max()) / 2.0)

    # "Pressured" = schema still substantially open. Use the upper half of THIS
    # trajectory's PE range as the cut; a pure stall never drains, so nearly every
    # exchange counts as pressured, which is the point -- friction there is
    # friction-while-work-remains.
    pressured = pressures >= pressure_cut
    regimes = [p.regime for p in points]

    # The pathological signature is friction that co-occurs with pressure. A
    # converging run keeps friction low across the whole high-PE stretch (any SI it
    # shows arrives only once PE has already fallen). A churning run keeps friction
    # high while PE is high. So the diagnostic scalar is simply mean friction over
    # the pressured exchanges: high = pathological, low = benign.
    friction_while_pressured = (
        float(frictions[pressured].mean()) if pressured.any() else 0.0
    )
    friction_when_drained = (
        float(frictions[~pressured].mean()) if (~pressured).any() else 0.0
    )

    return {
        "n": len(points),
        "churn_fraction": float(np.mean([r == CHURN for r in regimes])),
        "productive_fraction": float(np.mean([r == PRODUCTIVE for r in regimes])),
        "converged_fraction": float(np.mean([r == CONVERGED for r in regimes])),
        "quiet_incomplete_fraction": float(np.mean([r == QUIET_INCOMPLETE for r in regimes])),
        "friction_while_pressured": friction_while_pressured,
        "friction_when_drained": friction_when_drained,
        "terminal_regime": regimes[-1],
        # Headline scalar = churn fraction (see is_pathological). Scale-free,
        # transfers across substrates. friction_while_pressured is retained above
        # as a secondary, substrate-local diagnostic.
        "pathology_index": float(np.mean([r == CHURN for r in regimes])),
    }


# A trajectory is pathological when a sustained share of its exchanges sit in the
# CHURN corner: zero yield AND friction above the stall threshold. This is a pure
# count over regime labels, so unlike an absolute friction level it needs no
# PE-scale calibration and transfers across substrates unchanged. Measured: 0.000
# for every healthy run on both the BOIR and kinematic substrates, and 0.71-0.83
# for the two genuine stalls (BOIR ambiguous-recurrence, kinematic orbit-stall).
# "Sustained" = more than one isolated exchange, i.e. a run that gets stuck rather
# than passing through one high-SI blip.
CHURN_FRACTION_CUT = 0.15


def is_pathological(
    points: Sequence[PhasePoint], *, cut: float = CHURN_FRACTION_CUT
) -> bool:
    return phase_signature(points)["churn_fraction"] >= cut


# Kept for backward compatibility and for the plot caption; not the verdict.
PATHOLOGY_CUT = CHURN_FRACTION_CUT


def load_phase(path: str | Path, *, si_threshold: float = 0.20):
    """Read a trajectory file straight into phase points. No binding needed."""
    header, exchanges = load_trajectory(path)
    return header, trajectory_to_phase(exchanges, si_threshold=si_threshold)


def regime_counts(points: Sequence[PhasePoint]) -> dict[str, int]:
    counts = {regime: 0 for regime in REGIME_ORDER}
    for point in points:
        counts[point.regime] += 1
    return counts
