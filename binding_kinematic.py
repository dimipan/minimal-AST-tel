"""Dependency-free continuous kinematic manipulation binding for AST.

This is a deliberately small public analogue of a MetaWorld pick-and-place
binding. It uses deterministic NumPy kinematics so the full repository runs in
CI without MuJoCo. The schema and evidence representation remain compatible
with a later real-environment adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from substrate_binding import SubstrateBinding

KINEMATIC_DIMENSIONS = ("reach", "grasp", "lift", "place")
KINEMATIC_IMPORTANCE = {"reach": 0.7, "grasp": 0.9, "lift": 0.8, "place": 1.0}
KINEMATIC_DEPENDENCIES = {
    "grasp": ("reach",),
    "lift": ("grasp",),
    "place": ("lift",),
}

OBJECT_START = np.array([0.40, 0.00, 0.05], dtype=np.float64)
GOAL = np.array([0.72, 0.28, 0.05], dtype=np.float64)
TABLE_Z = 0.05


@dataclass(frozen=True)
class KinematicEvidence:
    evidence_id: str
    hand: np.ndarray
    obj: np.ndarray
    goal: np.ndarray
    gripper_closed: float
    grasped: bool
    target: str
    condition: str
    stall_gt: bool = False
    released_at_goal: bool = False


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def score_kinematics(evidence: KinematicEvidence) -> dict[str, float]:
    hand_object = float(np.linalg.norm(evidence.hand - evidence.obj))
    object_goal = float(np.linalg.norm(evidence.obj - evidence.goal))
    initial_goal_distance = float(np.linalg.norm(OBJECT_START - GOAL))

    reach = _clip01(1.0 - hand_object / 0.60)
    grasp = 1.0 if evidence.grasped else _clip01(
        evidence.gripper_closed * (1.0 - hand_object / 0.08)
    )
    lift = _clip01((evidence.obj[2] - TABLE_Z) / 0.20) if evidence.grasped else 0.0
    place_progress = _clip01(1.0 - object_goal / initial_goal_distance)
    place = max(place_progress if evidence.grasped else 0.0, 1.0 if evidence.released_at_goal else 0.0)
    return {"reach": reach, "grasp": grasp, "lift": lift, "place": place}


def embed_kinematics(evidence: KinematicEvidence) -> np.ndarray:
    """Thirteen-dimensional, unit-normalised kinematic evidence vector."""

    hand_object = float(np.linalg.norm(evidence.hand - evidence.obj))
    object_goal = float(np.linalg.norm(evidence.obj - evidence.goal))
    return np.concatenate(
        [
            evidence.hand,
            evidence.obj,
            evidence.goal,
            np.array(
                [
                    float(evidence.gripper_closed),
                    float(evidence.grasped),
                    hand_object,
                    object_goal,
                ],
                dtype=np.float64,
            ),
        ]
    )


def _frame(
    index: int,
    hand: np.ndarray,
    obj: np.ndarray,
    *,
    gripper: float,
    grasped: bool,
    target: str,
    condition: str,
    stall: bool = False,
    released: bool = False,
) -> KinematicEvidence:
    return KinematicEvidence(
        evidence_id=f"{condition}_{index:03d}",
        hand=np.asarray(hand, dtype=np.float64),
        obj=np.asarray(obj, dtype=np.float64),
        goal=GOAL.copy(),
        gripper_closed=float(gripper),
        grasped=bool(grasped),
        target=target,
        condition=condition,
        stall_gt=stall,
        released_at_goal=released,
    )


def _approach(condition: str, start_index: int = 1) -> list[KinematicEvidence]:
    start = np.array([0.05, -0.25, 0.22])
    end = OBJECT_START + np.array([0.0, 0.0, 0.035])
    frames = []
    for offset, alpha in enumerate(np.linspace(0.0, 1.0, 6)):
        hand = start * (1.0 - alpha) + end * alpha
        frames.append(
            _frame(
                start_index + offset,
                hand,
                OBJECT_START,
                gripper=0.0,
                grasped=False,
                target="reach",
                condition=condition,
            )
        )
    return frames


def clean_sequence(condition: str = "clean") -> list[KinematicEvidence]:
    frames = _approach(condition)
    index = len(frames) + 1
    grasp_hand = OBJECT_START + np.array([0.0, 0.0, 0.025])
    frames.append(
        _frame(index, grasp_hand, OBJECT_START, gripper=0.55, grasped=False, target="grasp", condition=condition)
    )
    index += 1
    frames.append(
        _frame(index, grasp_hand, OBJECT_START, gripper=1.0, grasped=True, target="grasp", condition=condition)
    )
    index += 1

    lifted = OBJECT_START.copy()
    for alpha in np.linspace(0.25, 1.0, 4):
        lifted = OBJECT_START + np.array([0.0, 0.0, 0.20 * alpha])
        frames.append(
            _frame(index, lifted + np.array([0.0, 0.0, 0.025]), lifted, gripper=1.0, grasped=True, target="lift", condition=condition)
        )
        index += 1

    move_start = lifted.copy()
    goal_above = GOAL + np.array([0.0, 0.0, 0.20])
    for alpha in np.linspace(0.2, 1.0, 6):
        obj = move_start * (1.0 - alpha) + goal_above * alpha
        frames.append(
            _frame(index, obj + np.array([0.0, 0.0, 0.025]), obj, gripper=1.0, grasped=True, target="place", condition=condition)
        )
        index += 1

    for alpha in np.linspace(0.25, 1.0, 4):
        obj = goal_above * (1.0 - alpha) + GOAL * alpha
        frames.append(
            _frame(index, obj + np.array([0.0, 0.0, 0.025]), obj, gripper=1.0, grasped=True, target="place", condition=condition)
        )
        index += 1
    frames.append(
        _frame(index, GOAL + np.array([0.0, 0.0, 0.08]), GOAL, gripper=0.0, grasped=False, target="place", condition=condition, released=True)
    )
    return frames


def orbit_stall_sequence() -> list[KinematicEvidence]:
    condition = "orbit_stall"
    frames = _approach(condition)
    index = len(frames) + 1
    radius = 0.075
    for angle in np.linspace(0.0, 4.0 * np.pi, 18, endpoint=False):
        hand = OBJECT_START + np.array([radius * np.cos(angle), radius * np.sin(angle), 0.04])
        frames.append(
            _frame(
                index,
                hand,
                OBJECT_START,
                gripper=0.0,
                grasped=False,
                target="grasp",
                condition=condition,
                stall=True,
            )
        )
        index += 1
    return frames


def saturated_hold_sequence() -> list[KinematicEvidence]:
    condition = "saturated_hold"
    frames = clean_sequence(condition)
    index = len(frames) + 1
    for _ in range(8):
        frames.append(
            _frame(
                index,
                GOAL + np.array([0.0, 0.0, 0.08]),
                GOAL,
                gripper=0.0,
                grasped=False,
                target="place",
                condition=condition,
                stall=False,
                released=True,
            )
        )
        index += 1
    return frames


KINEMATIC_SCENARIOS = {
    "clean": clean_sequence(),
    "orbit_stall": orbit_stall_sequence(),
    "saturated_hold": saturated_hold_sequence(),
}


def make_kinematic_binding() -> SubstrateBinding:
    return SubstrateBinding(
        name="synthetic-kinematic-pick-place",
        schema=KINEMATIC_DIMENSIONS,
        importance=KINEMATIC_IMPORTANCE,
        dependencies=KINEMATIC_DEPENDENCIES,
        scorer=score_kinematics,
        embedder=embed_kinematics,
        target_selector=lambda evidence: evidence.target,
        evidence_ref=lambda evidence: {
            "id": evidence.evidence_id,
            "hand": np.round(evidence.hand, 5).tolist(),
            "object": np.round(evidence.obj, 5).tolist(),
            "goal": np.round(evidence.goal, 5).tolist(),
            "gripper_closed": evidence.gripper_closed,
            "grasped": evidence.grasped,
        },
        ground_truth=lambda evidence: {
            "condition": evidence.condition,
            "stall": evidence.stall_gt,
            "released_at_goal": evidence.released_at_goal,
        },
    )


def iter_all_frames() -> Iterable[KinematicEvidence]:
    for frames in KINEMATIC_SCENARIOS.values():
        yield from frames
