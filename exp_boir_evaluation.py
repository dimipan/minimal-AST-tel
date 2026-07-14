"""Evaluate AST over a recursive Bayesian intent estimator (BOIR).

The monitored plant here is not a task -- it is another inference process. This
is the substrate that makes the substrate-agnosticism claim non-trivial: the same
frozen equations that watch a witness interview and a robot arm also watch the
epistemic state of a Bayesian estimator.

Two results this script is designed to surface, both of which are boundary
conditions rather than successes:

1. THE SCORER MUST BE NON-MONOTONE IN THE ESTIMATOR'S STATE.
   AST integrates monotonically. A posterior can un-resolve. Decisiveness
   (1 - H_b(p)) collapses at p = 0.5, so churn cannot ratchet it and the stall
   fires. Veridical resolvedness is monotone in p, so a transient swing is locked
   in by the running maximum and SI is suppressed in exactly the scenario built to
   produce a stall. Measured: AUC 1.000 vs 0.854.

2. A TRUTH-FREE FRICTION CHANNEL CANNOT CERTIFY CORRECTNESS.
   In `confidently_wrong` the estimator adjudicates the hypothesis space cleanly
   and efficiently -- and lands on the wrong goal in every single window. SI is
   LOW, and SI is RIGHT to be low: there is no friction. The estimator is not
   stuck; it is confident and mistaken. AST can tell you whether an inference
   process is still learning. It cannot tell you whether what it learned is true.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bayesian_intent import RecursiveBayesianIntentEstimator
from binding_boir import WINDOW, run_boir_ast
from evaluation_utils import (
    detector_metrics,
    format_sweep,
    separation_metrics,
    threshold_sweep,
)
from plotting import (
    plot_evidence_geometry,
    plot_regime_portrait,
    plot_scenario_comparison,
    plot_telemetry_trajectory,
    plot_trajectory,
    substrate_limits,
)
from synthetic_intent_trajectories import GOAL_NAMES, SCENARIO_BUILDERS, all_scenarios

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "calibration_manifest.json").read_text(encoding="utf-8"))


def _fmt(value, width=7):
    return f"{'--':>{width}}" if value is None else f"{value:{width}.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIO_BUILDERS),
                        default=list(SCENARIO_BUILDERS))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/boir"))
    parser.add_argument("--scorer", choices=("decisiveness", "veridical"),
                        default="decisiveness",
                        help="decisiveness is deployable and is the default; "
                             "veridical requires ground truth and ratchets under churn")
    parser.add_argument("--window", type=int, default=WINDOW,
                        help="estimator ticks per AST exchange")
    parser.add_argument("--si-threshold", type=float, default=0.20)
    parser.add_argument("--delta", type=float, default=0.2,
                        help="BOIR transition-model Delta")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.scorer == "veridical":
        print(
            "\n  WARNING: the veridical scorer is monotone in the posterior, so the\n"
            "  running maximum locks in transient swings and suppresses SI under\n"
            "  churn. It is a diagnostic, not the headline. See binding_boir.py.\n"
        )

    provenance = {
        "binding": f"synthetic-boir-intent-{args.scorer}",
        "monitored_process": "recursive Bayesian intent estimator (BOIR)",
        "representation": f"{4 * len(GOAL_NAMES) + 6}-d inference-dynamics signature",
        "scorer": args.scorer,
        "window_ticks": args.window,
        "n_goals": len(GOAL_NAMES),
        "boir_delta": args.delta,
        "ast_resets_per_epoch": True,
        "boir_runs_continuously": True,
        "calibration": MANIFEST["manifest_version"],
    }
    print("Provenance")
    for key, value in provenance.items():
        print(f"  {key:24s} {value}")

    scenarios = {name: all_scenarios()[name] for name in args.scenarios}
    runs, embeddings, epistemic = {}, {}, {}
    from binding_boir import build_windows, make_boir_binding
    binding = make_boir_binding(scorer=args.scorer)

    for name, trajectory in scenarios.items():
        estimator = RecursiveBayesianIntentEstimator(
            n_goals=trajectory.n_goals, delta=args.delta
        )
        by_epoch = run_boir_ast(
            trajectory, estimator=estimator, scorer=args.scorer,
            window=args.window, output_dir=args.output_dir,
        )
        windows = build_windows(
            trajectory,
            RecursiveBayesianIntentEstimator(n_goals=trajectory.n_goals, delta=args.delta),
            window=args.window,
        )
        for epoch, records in by_epoch.items():
            key = f"{name}/e{epoch}" if len(by_epoch) > 1 else name
            runs[key] = records
            embeddings[key] = np.stack([binding.embed(w) for w in windows[epoch]])
            correct = [r["ground_truth"]["map_correct"] for r in records]
            epistemic[key] = {
                "true_goal": records[0]["ground_truth"]["true_goal"],
                "map_correct_windows": f"{sum(correct)}/{len(correct)}",
                "map_accuracy": sum(correct) / len(correct),
                "final_p_true": records[-1]["ground_truth"]["p_true"],
                "final_knowledge_mean": records[-1]["knowledge_mean"],
                "mean_si": float(np.mean([r["si"] for r in records])),
            }

    limits = substrate_limits(runs, si_threshold=args.si_threshold)

    # -- headline ------------------------------------------------------------
    separation = {name: separation_metrics(records) for name, records in runs.items()}
    pooled = separation_metrics([r for records in runs.values() for r in records])
    print("\nHeadline (threshold-free)")
    print(f"  {'condition':26s} {'AUC':>7s} {'meanSI stall':>13s} {'meanSI clean':>13s} "
          f"{'rank-perfect':>13s}")
    for name, m in separation.items():
        print(f"  {name:26s} {_fmt(m['auc'])} {_fmt(m['mean_si_stall'], 13)} "
              f"{_fmt(m['mean_si_clean'], 13)} {str(m['perfect_rank_separation']):>13s}")
    print(f"  {'POOLED':26s} {_fmt(pooled['auc'])} {_fmt(pooled['mean_si_stall'], 13)} "
          f"{_fmt(pooled['mean_si_clean'], 13)} {str(pooled['perfect_rank_separation']):>13s}")

    # -- the epistemic table: what the friction channel CANNOT see ------------
    print("\nEstimator correctness (what SI cannot see)")
    print(f"  {'condition':26s} {'true goal':>10s} {'MAP correct':>12s} {'p(true)':>8s} "
          f"{'completeness':>12s} {'mean SI':>8s}")
    for name, e in epistemic.items():
        print(f"  {name:26s} {e['true_goal']:>10s} {e['map_correct_windows']:>12s} "
              f"{e['final_p_true']:8.3f} {e['final_knowledge_mean']:12.3f} "
              f"{e['mean_si']:8.3f}")
    if "confidently_wrong" in epistemic:
        e = epistemic["confidently_wrong"]
        print(
            f"\n  Read the confidently_wrong row. The estimator adjudicated the "
            f"hypothesis\n  space to completeness {e['final_knowledge_mean']:.3f} with "
            f"mean SI {e['mean_si']:.3f} -- no friction, no stall --\n  and its MAP "
            f"hypothesis was wrong in {e['map_correct_windows'].split('/')[1]} of "
            f"{e['map_correct_windows'].split('/')[1]} windows.\n"
            f"  SI is CORRECT to be low: the estimator is not stuck. It is confident\n"
            f"  and mistaken. A truth-free friction channel cannot distinguish those."
        )

    detector = {
        name: detector_metrics(records, threshold=args.si_threshold)
        for name, records in runs.items()
    }
    sweep = threshold_sweep(runs)
    print(f"\nPooled threshold sweep (theta is NOT fitted for this binding)")
    print(format_sweep(sweep))

    # -- figures -------------------------------------------------------------
    for name, records in runs.items():
        label = f"BOIR intent — {name.replace('_', ' ')}"
        safe = name.replace("/", "_")
        plot_trajectory(records, GOAL_NAMES, args.output_dir / f"{safe}.png",
                        title=label, limits=limits)
        plot_telemetry_trajectory(
            records, args.output_dir / f"{safe}_telemetry_trajectory.png",
            title=label, limits=limits,
        )
    plot_scenario_comparison(runs, args.output_dir / "scenario_comparison.png",
                             title=f"BOIR intent substrate ({args.scorer})", limits=limits)
    plot_regime_portrait(runs, args.output_dir / "regime_portrait.png",
                         title=f"BOIR intent substrate ({args.scorer})", limits=limits)
    _p, explained = plot_evidence_geometry(
        embeddings, runs, args.output_dir / "evidence_geometry.png",
        title=f"BOIR inference dynamics ({args.scorer})",
    )
    print(f"\nEvidence geometry: 2 PCA components explain {explained:.1%} of variance")

    (args.output_dir / "summary.json").write_text(
        json.dumps({
            "provenance": provenance,
            "headline_threshold_free": separation,
            "pooled": pooled,
            "estimator_correctness": epistemic,
            "fixed_threshold_operating_point": detector,
            "threshold_sweep": sweep,
            "pca_explained_variance_2c": explained,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    keys = list(next(iter(separation.values())).keys())
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", *keys])
        writer.writeheader()
        for name, metrics in separation.items():
            writer.writerow({"condition": name, **metrics})

    print(f"\nwrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
