"""Deterministic synthetic observation streams for the BOIR substrate.

No robot, no simulator, no logs, no extracted data. Each scenario is a pair of
NumPy arrays

    angles.shape == (T, N)      degrees, lower = more consistent with that goal
    paths.shape  == (T, N)      metres,  lower = more consistent with that goal

generated in closed form from a seed. Everything downstream -- the estimator, the
binding, the telemetry -- is a deterministic function of these two arrays.

Four scenarios, chosen to span the regimes the monitor claims to distinguish:

  clean_convergence     evidence steadily favours one goal
  ambiguous_recurrence  two goals stay equally plausible; the posterior churns
  intent_switch         the latent goal changes half way through
  confidently_wrong     evidence consistently favours a goal that is NOT the
                        true one -- the estimator resolves, and resolves wrongly
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_GOALS = 4
GOAL_NAMES = tuple(f"goal_{i + 1}" for i in range(N_GOALS))


@dataclass(frozen=True)
class IntentTrajectory:
    """One synthetic observation stream plus its latent truth."""

    name: str
    angles: np.ndarray          # (T, N)
    paths: np.ndarray           # (T, N)
    epoch_boundaries: tuple[tuple[int, int], ...]   # half-open [start, end)
    true_goal_by_epoch: tuple[int, ...]             # index into GOAL_NAMES
    description: str
    # Constructed ground truth, DECLARED BY THE SCENARIO -- not derived from a
    # rule applied to the output. Which epochs are stalls by design, i.e. the
    # hypothesis space is never adjudicated no matter how much evidence arrives.
    stall_epochs: tuple[int, ...] = ()

    @property
    def n_ticks(self) -> int:
        return int(self.angles.shape[0])

    @property
    def n_goals(self) -> int:
        return int(self.angles.shape[1])

    def epoch_of(self, tick: int) -> int:
        for index, (start, end) in enumerate(self.epoch_boundaries):
            if start <= tick < end:
                return index
        raise IndexError(f"tick {tick} lies outside every epoch")


def _noise(rng: np.random.Generator, shape, scale: float) -> np.ndarray:
    return rng.normal(0.0, scale, size=shape)


def _clip_angle(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 180.0)


def _clip_path(l: np.ndarray) -> np.ndarray:
    return np.clip(l, 0.5, 25.0)


def make_clean_convergence(n_ticks: int = 60, true_goal: int = 0, seed: int = 7) -> IntentTrajectory:
    """Angle and path to the true goal fall steadily; competitors stay far."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_ticks)[:, None]

    angles = np.full((n_ticks, N_GOALS), 110.0)
    paths = np.full((n_ticks, N_GOALS), 18.0)
    for goal in range(N_GOALS):
        if goal == true_goal:
            angles[:, goal] = (85.0 * (1.0 - t) + 6.0 * t).ravel()
            paths[:, goal] = (20.0 * (1.0 - t) + 2.0 * t).ravel()
        else:
            drift = 0.35 * (goal + 1)
            angles[:, goal] = (95.0 + 25.0 * drift * t).ravel()
            paths[:, goal] = (16.0 + 6.0 * drift * t).ravel()

    angles = _clip_angle(angles + _noise(rng, angles.shape, 2.0))
    paths = _clip_path(paths + _noise(rng, paths.shape, 0.4))
    return IntentTrajectory(
        name="clean_convergence",
        angles=angles,
        paths=paths,
        epoch_boundaries=((0, n_ticks),),
        true_goal_by_epoch=(true_goal,),
        description="Evidence steadily favours one goal; the posterior converges.",
        stall_epochs=(),
    )


def make_ambiguous_recurrence(
    n_ticks: int = 60, true_goal: int = 0, rival: int = 1, seed: int = 11
) -> IntentTrajectory:
    """Two goals remain equally plausible; the evidence oscillates between them.

    This is the principal synthetic stall. The operator keeps producing data, the
    estimator keeps processing it, and the hypothesis space is no more adjudicated
    at tick 60 than at tick 10. Neither goal is ever eliminated.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_ticks)
    oscillation = 16.0 * np.sin(2.0 * np.pi * t / 11.0)

    angles = np.full((n_ticks, N_GOALS), 120.0)
    paths = np.full((n_ticks, N_GOALS), 20.0)

    angles[:, true_goal] = 46.0 + oscillation
    angles[:, rival] = 46.0 - oscillation
    paths[:, true_goal] = 11.0 + 0.10 * oscillation
    paths[:, rival] = 11.0 - 0.10 * oscillation

    for goal in range(N_GOALS):
        if goal in (true_goal, rival):
            continue
        angles[:, goal] = 130.0 + 4.0 * np.sin(2.0 * np.pi * t / 7.0 + goal)
        paths[:, goal] = 21.0

    angles = _clip_angle(angles + _noise(rng, angles.shape, 1.5))
    paths = _clip_path(paths + _noise(rng, paths.shape, 0.3))
    return IntentTrajectory(
        name="ambiguous_recurrence",
        angles=angles,
        paths=paths,
        epoch_boundaries=((0, n_ticks),),
        true_goal_by_epoch=(true_goal,),
        description=(
            "Two goals stay equally plausible; the posterior churns without ever "
            "adjudicating. Evidence keeps arriving; intent is never resolved."
        ),
        # The WHOLE epoch is a stall by construction. The oscillation never lets
        # either contested hypothesis be confirmed or eliminated.
        stall_epochs=(0,),
    )


def make_intent_switch(
    n_ticks: int = 80, first_goal: int = 0, second_goal: int = 2, seed: int = 13
) -> IntentTrajectory:
    """The latent goal changes at the half-way point.

    BOIR runs continuously across the boundary -- its carry-over and lag are the
    phenomenon. AST resets, because the epistemic task has changed.
    """
    rng = np.random.default_rng(seed)
    half = n_ticks // 2
    angles = np.full((n_ticks, N_GOALS), 115.0)
    paths = np.full((n_ticks, N_GOALS), 18.0)

    for start, end, target in ((0, half, first_goal), (half, n_ticks, second_goal)):
        span = end - start
        t = np.linspace(0.0, 1.0, span)
        for goal in range(N_GOALS):
            if goal == target:
                angles[start:end, goal] = 80.0 * (1.0 - t) + 8.0 * t
                paths[start:end, goal] = 19.0 * (1.0 - t) + 3.0 * t
            else:
                angles[start:end, goal] = 100.0 + 20.0 * t
                paths[start:end, goal] = 17.0 + 4.0 * t

    angles = _clip_angle(angles + _noise(rng, angles.shape, 2.0))
    paths = _clip_path(paths + _noise(rng, paths.shape, 0.4))
    return IntentTrajectory(
        name="intent_switch",
        angles=angles,
        paths=paths,
        epoch_boundaries=((0, half), (half, n_ticks)),
        true_goal_by_epoch=(first_goal, second_goal),
        description=(
            "The latent goal changes at the midpoint. BOIR is continuous across the "
            "boundary; AST resets, because the epistemic task has changed."
        ),
        stall_epochs=(),   # both epochs resolve; the interest is in lag, not stall
    )


def make_confidently_wrong(
    n_ticks: int = 60, true_goal: int = 0, apparent_goal: int = 1, seed: int = 17
) -> IntentTrajectory:
    """The estimator resolves cleanly -- onto the wrong goal.

    The negative control for the deployable scorer. Decisiveness will report the
    hypothesis space as resolved; the veridical scorer will report that it resolved
    incorrectly. AST can see whether an estimator is still learning. It cannot see
    whether what it learned is true.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_ticks)[:, None]

    angles = np.full((n_ticks, N_GOALS), 115.0)
    paths = np.full((n_ticks, N_GOALS), 18.0)
    for goal in range(N_GOALS):
        if goal == apparent_goal:
            angles[:, goal] = (82.0 * (1.0 - t) + 7.0 * t).ravel()
            paths[:, goal] = (19.0 * (1.0 - t) + 2.5 * t).ravel()
        else:
            drift = 0.3 * (goal + 1)
            angles[:, goal] = (98.0 + 24.0 * drift * t).ravel()
            paths[:, goal] = (16.5 + 5.0 * drift * t).ravel()

    angles = _clip_angle(angles + _noise(rng, angles.shape, 2.0))
    paths = _clip_path(paths + _noise(rng, paths.shape, 0.4))
    return IntentTrajectory(
        name="confidently_wrong",
        angles=angles,
        paths=paths,
        epoch_boundaries=((0, n_ticks),),
        true_goal_by_epoch=(true_goal,),
        description=(
            "Observations consistently favour a goal that is not the true one. The "
            "estimator resolves confidently and incorrectly."
        ),
        # NOT a stall, and that is the entire point. The estimator adjudicates the
        # hypothesis space cleanly and efficiently. SI is CORRECT to stay low. The
        # answer is simply wrong, and no truth-free friction channel can see that.
        stall_epochs=(),
    )


SCENARIO_BUILDERS = {
    "clean_convergence": make_clean_convergence,
    "ambiguous_recurrence": make_ambiguous_recurrence,
    "intent_switch": make_intent_switch,
    "confidently_wrong": make_confidently_wrong,
}


def all_scenarios() -> dict[str, IntentTrajectory]:
    return {name: build() for name, build in SCENARIO_BUILDERS.items()}
