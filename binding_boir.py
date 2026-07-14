"""Substrate binding: AST over a recursive Bayesian intent estimator (BOIR).

The monitored plant is not a task. It is another inference process. AST asks:

    Is incoming evidence still resolving operator intent, or is the estimator
    processing recurrent evidence without epistemic progress?

    B = (M, Y, S, e)

    M : goal HYPOTHESES {h_1..h_N}. Dimension i = "hypothesis i adjudicated",
        by confirmation OR elimination. Ruling a goal out is acquisition, so
        every dimension is reachable and (0, 0) exists for each epoch.
    Y : a window of K estimator ticks -- posterior dynamics plus the angle and
        path observations driving them.
    S : decisiveness, s_i = 1 - H_b(p_i). Deployable: it needs no ground truth.
    e : unit-norm inference-dynamics signature of the window.
    i_t : the MAP hypothesis, majority-voted over the window.


Why decisiveness is the DEFAULT scorer, and veridical is not
------------------------------------------------------------
This was settled empirically, not by preference. AST integrates monotonically,
upsilon_i(t) = max(upsilon_i(t-1), s_i(t)), and a posterior can un-resolve. So
the question is whether the running maximum survives posterior churn.

Measured on `ambiguous_recurrence`, on the two contested hypotheses:

    scorer          final upsilon      saturation weight (1 - upsilon)
    decisiveness    0.367 / 0.310      0.661
    veridical       0.745 / 0.774      0.241

Decisiveness is NON-monotone in p: it collapses to zero at p = 0.5, exactly where
a contested hypothesis lives. Churn therefore cannot ratchet it -- the running
maximum plateaus early and the saturation weight stays high, so SI keeps firing
on the unresolved hypotheses. That is the intended behaviour.

Veridical resolvedness (p_i for the true goal, 1 - p_i for the others) IS monotone
in p. A transient swing gets locked in by the running maximum, upsilon records a
peak the estimator never sustained, and the (1 - upsilon) term suppresses SI by
roughly 2.7x in precisely the scenario built to produce a stall.

So veridical scoring is NOT fed to the monotone integrator. It is computed and
exported as a ground-truth ANNOTATION (p_true, map_correct) for evaluation, which
is what it was always for. `--scorer veridical` remains available as a diagnostic
and prints a warning; do not report it as the headline.

This is a boundary condition of AST worth stating plainly: monotone integration
is appropriate when acquisition is irreversible, and a process that can LOSE
ground needs a scorer that is bounded away from 1 while its state is contested.


Epochs
------
Within an epoch the latent goal is fixed, so evidence accumulation about it is a
legitimately monotone process. Across an intent change it is not. So:

    BOIR runs CONTINUOUSLY across the whole trajectory -- its carry-over and lag
    are the phenomenon under diagnosis.
    AST RESETS at each boundary -- one run_binding session per epoch.

That needs no change to the shared runner: each epoch is simply its own
ast-tau-v1.1 session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from bayesian_intent import (
    RecursiveBayesianIntentEstimator,
    binary_entropy,
    kl_divergence,
)
from substrate_binding import SubstrateBinding, run_binding
from synthetic_intent_trajectories import GOAL_NAMES, IntentTrajectory

WINDOW = 5          # K estimator ticks per AST exchange


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InferenceWindow:
    """One AST exchange: K ticks of the estimator's epistemic process."""

    window_id: str
    epoch: int
    t: int
    posteriors: np.ndarray      # (K, N)
    prior_posterior: np.ndarray  # (N,) the posterior entering the window
    angles: np.ndarray          # (K, N)
    paths: np.ndarray           # (K, N)
    map_index: int
    true_goal: int
    condition: str
    stall_gt: bool

    @property
    def n_goals(self) -> int:
        return int(self.posteriors.shape[1])


def decisiveness_scores(window: InferenceWindow) -> dict[str, float]:
    """S(y_t): deployable, truth-free. s_i = 1 - H_b(p_i) at the window's end.

    A hypothesis is adjudicated when it is either strongly confirmed (p -> 1) or
    strongly eliminated (p -> 0). It is unadjudicated at p = 0.5. This is exactly
    the shape that makes it safe under the monotone integrator.
    """
    final = window.posteriors[-1]
    scores = 1.0 - binary_entropy(final)
    return {
        GOAL_NAMES[i]: float(np.clip(scores[i], 0.0, 1.0))
        for i in range(window.n_goals)
    }


def veridical_scores(window: InferenceWindow) -> dict[str, float]:
    """Evaluation-only. Requires ground truth, and ratchets under churn -- see
    the module docstring. Exposed for diagnosis, not for headline reporting."""
    final = window.posteriors[-1]
    scores = np.where(
        np.arange(window.n_goals) == window.true_goal, final, 1.0 - final
    )
    return {
        GOAL_NAMES[i]: float(np.clip(scores[i], 0.0, 1.0))
        for i in range(window.n_goals)
    }


def embed_inference_window(window: InferenceWindow) -> np.ndarray:
    """e(y_t): unit-norm inference-dynamics signature.

    Both observation sources appear explicitly. The original research embedding
    carried angle dynamics on the MAP goal but not path dynamics, which left half
    the evidence driving the estimator invisible to the friction channel.

    For N goals the vector is 4N + 6 wide (22 at N = 4), comfortably above the
    number of windows any single schema dimension can accumulate. That matters:
    SI's orthogonal novelty is computed against the span of prior evidence FOR
    THAT DIMENSION, so an embedding narrower than the per-dimension log length
    would let the subspace reach full rank, drive eta to zero, and collapse the
    geometry out of the signal.
    """
    posteriors = window.posteriors
    final = posteriors[-1]
    delta = final - window.prior_posterior

    entropies = np.array([
        float(-np.sum(np.clip(p, 1e-300, 1.0) * np.log(np.clip(p, 1e-300, 1.0))))
        for p in posteriors
    ])
    entropy_slope = float(entropies[-1] - entropies[0])
    innovations = [
        kl_divergence(posteriors[k], posteriors[k - 1]) for k in range(1, len(posteriors))
    ]
    mean_kl = float(np.mean(innovations)) if innovations else 0.0
    maps = posteriors.argmax(axis=1)
    switches = float((np.diff(maps) != 0).sum())

    m = window.map_index
    map_angles = window.angles[:, m]
    map_paths = window.paths[:, m]

    return np.concatenate([
        posteriors.mean(axis=0),                        # N  posterior mean
        delta,                                          # N  posterior change
        window.angles.mean(axis=0) / 180.0,             # N  angle evidence
        window.paths.mean(axis=0) / 25.0,               # N  path evidence
        np.array([
            float(entropies.mean()),                    # uncertainty level
            entropy_slope,                              # uncertainty trend
            mean_kl,                                    # information flow
            switches,                                   # estimate churn
            float(map_angles[-1] - map_angles[0]) / 180.0,   # MAP angle change
            float(map_paths[-1] - map_paths[0]) / 25.0,      # MAP path change
        ]),
    ])


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def build_windows(
    trajectory: IntentTrajectory,
    estimator: RecursiveBayesianIntentEstimator,
    *,
    window: int = WINDOW,
) -> dict[int, list[InferenceWindow]]:
    """Replay BOIR over the whole trajectory, then cut it into per-epoch windows.

    The estimator is NOT reset at an epoch boundary. AST is (by starting a new
    session per epoch). That asymmetry is the design.
    """
    estimator.reset()
    posteriors = estimator.run(trajectory.angles, trajectory.paths)

    by_epoch: dict[int, list[InferenceWindow]] = {}
    for epoch, (start, end) in enumerate(trajectory.epoch_boundaries):
        true_goal = trajectory.true_goal_by_epoch[epoch]
        windows: list[InferenceWindow] = []
        for index, w0 in enumerate(range(start, end - window + 1, window), start=1):
            span = slice(w0, w0 + window)
            block = posteriors[span]
            prior = posteriors[w0 - 1] if w0 > 0 else np.full(
                trajectory.n_goals, 1.0 / trajectory.n_goals
            )
            maps = block.argmax(axis=1)
            map_index = int(np.bincount(maps, minlength=trajectory.n_goals).argmax())

            # Ground truth is DECLARED BY THE SCENARIO, not derived from a rule
            # applied to the monitor's own inputs -- deriving it would make the
            # evaluation partly circular. An epoch designed as a stall labels every
            # one of its windows a stall.
            #
            # Exchange 1 is never labelled: SI is identically zero on the first
            # exchange of any session (no prior subspace exists to be recurrent
            # against), so labelling it would bake in a guaranteed false negative.
            stall = bool(epoch in trajectory.stall_epochs and index >= 2)

            windows.append(InferenceWindow(
                window_id=f"{trajectory.name}_e{epoch}_w{index:03d}",
                epoch=epoch,
                t=index,
                posteriors=block,
                prior_posterior=prior,
                angles=trajectory.angles[span],
                paths=trajectory.paths[span],
                map_index=map_index,
                true_goal=true_goal,
                condition=trajectory.name,
                stall_gt=stall,
            ))
        by_epoch[epoch] = windows
    return by_epoch


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------
def make_boir_binding(*, scorer: str = "decisiveness") -> SubstrateBinding:
    if scorer not in ("decisiveness", "veridical"):
        raise ValueError("scorer must be 'decisiveness' or 'veridical'")
    score_fn = decisiveness_scores if scorer == "decisiveness" else veridical_scores

    return SubstrateBinding(
        name=f"synthetic-boir-intent-{scorer}",
        schema=GOAL_NAMES,
        importance={name: 1.0 for name in GOAL_NAMES},
        dependencies={},
        scorer=score_fn,
        embedder=embed_inference_window,
        target_selector=lambda w: GOAL_NAMES[w.map_index],
        evidence_ref=lambda w: {
            "id": w.window_id,
            "epoch": w.epoch,
            "posterior": np.round(w.posteriors[-1], 5).tolist(),
            "map": GOAL_NAMES[w.map_index],
        },
        ground_truth=lambda w: {
            "condition": w.condition,
            "stall": w.stall_gt,
            "epoch": w.epoch,
            "true_goal": GOAL_NAMES[w.true_goal],
            "map_correct": bool(w.map_index == w.true_goal),
            # Veridical resolvedness, exported as an ANNOTATION rather than fed to
            # the monotone integrator. This is what separates confidently-wrong
            # from correctly-resolved, and the deployable scorer cannot see it.
            "p_true": float(w.posteriors[-1][w.true_goal]),
        },
    )


def run_boir_ast(
    trajectory: IntentTrajectory,
    *,
    estimator: RecursiveBayesianIntentEstimator | None = None,
    scorer: str = "decisiveness",
    window: int = WINDOW,
    output_dir=None,
) -> dict[int, list[dict]]:
    """Run BOIR, then AST over BOIR. One AST session per intent epoch."""
    estimator = estimator or RecursiveBayesianIntentEstimator(n_goals=trajectory.n_goals)
    binding = make_boir_binding(scorer=scorer)
    windows = build_windows(trajectory, estimator, window=window)

    records: dict[int, list[dict]] = {}
    for epoch, stream in windows.items():
        path = None
        if output_dir is not None:
            path = output_dir / f"{trajectory.name}_epoch{epoch}.jsonl"
        records[epoch] = run_binding(
            binding,
            stream,
            output_path=path,
            session_id=f"{trajectory.name}_epoch{epoch}",
            meta={
                "synthetic": True,
                "modality": "estimator-epistemic-state",
                "monitored_process": "recursive Bayesian intent estimator (BOIR)",
                "scorer": scorer,
                "window_ticks": window,
                "epoch": epoch,
                "true_goal": GOAL_NAMES[trajectory.true_goal_by_epoch[epoch]],
                "ast_resets_per_epoch": True,
                "boir_runs_continuously": True,
            },
        )
    return records
