"""Evaluate AST on the dependency-free kinematic manipulation substrate.

Unlike the linguistic substrate, this binding's evidence representation IS the
declared numeric state vector, not a learned embedding. Its SI scale is therefore
stable, and the threshold declared in calibration_manifest.json is usable. Both
the threshold-free headline and the fixed-threshold table are reported, so the
two substrates can be read side by side.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from binding_kinematic import (
    KINEMATIC_DIMENSIONS,
    KINEMATIC_SCENARIOS,
    make_kinematic_binding,
)
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
from substrate_binding import run_binding

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "calibration_manifest.json").read_text(encoding="utf-8"))


def _fmt(value, width=7):
    return f"{'--':>{width}}" if value is None else f"{value:{width}.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=sorted(KINEMATIC_SCENARIOS),
                        default=list(KINEMATIC_SCENARIOS))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kinematic"))
    parser.add_argument("--si-threshold", type=float, default=None,
                        help="defaults to the threshold declared for this binding")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    binding = make_kinematic_binding()
    declared = (
        MANIFEST["per_binding_calibration"]
        .get(binding.name, {})
        .get("offline_detection_threshold")
    )
    threshold = args.si_threshold if args.si_threshold is not None else (declared or 0.20)

    provenance = {
        "binding": binding.name,
        "representation": "declared numeric state vector (13-d), not a learned embedding",
        "calibration": MANIFEST["manifest_version"],
        "declared_threshold": declared,
        "threshold_used": threshold,
    }
    print("\nRepresentation provenance")
    for key, value in provenance.items():
        print(f"  {key:22s} {value}")

    runs, embeddings = {}, {}
    for scenario in args.scenarios:
        runs[scenario] = run_binding(
            binding,
            KINEMATIC_SCENARIOS[scenario],
            output_path=args.output_dir / f"{scenario}.jsonl",
            session_id=scenario,
            meta={"synthetic": True, "modality": "continuous-kinematic", **provenance},
        )
        embeddings[scenario] = np.stack(
            [binding.embed(e) for e in KINEMATIC_SCENARIOS[scenario]]
        )

    limits = substrate_limits(runs, si_threshold=threshold)

    separation = {name: separation_metrics(records) for name, records in runs.items()}
    print("\nHeadline (threshold-free)")
    print(f"  {'condition':26s} {'AUC':>7s} {'meanSI stall':>13s} "
          f"{'meanSI clean':>13s} {'separation':>11s} {'rank-perfect':>13s}")
    for name, m in separation.items():
        print(f"  {name:26s} {_fmt(m['auc'])} {_fmt(m['mean_si_stall'], 13)} "
              f"{_fmt(m['mean_si_clean'], 13)} {_fmt(m['si_separation'], 11)} "
              f"{str(m['perfect_rank_separation']):>13s}")

    detector = {
        name: detector_metrics(records, threshold=threshold)
        for name, records in runs.items()
    }
    sweep = threshold_sweep(runs)
    print(f"\nFixed-threshold operating point (theta = {threshold:.2f})")
    for name, m in detector.items():
        print(f"  {name:26s} recall={m['recall']} precision={m['precision']} "
              f"fpr={m['false_positive_rate']}")
    print("\nPooled threshold sweep")
    print(format_sweep(sweep))

    for name, records in runs.items():
        label = f"Kinematic pick-and-place — {name.replace('_', ' ')}"
        plot_trajectory(records, KINEMATIC_DIMENSIONS, args.output_dir / f"{name}.png",
                        title=label, limits=limits)
        plot_telemetry_trajectory(
            records, args.output_dir / f"{name}_telemetry_trajectory.png",
            title=label, limits=limits,
        )
    plot_scenario_comparison(runs, args.output_dir / "scenario_comparison.png",
                             title="Synthetic kinematic substrate", limits=limits)
    plot_regime_portrait(runs, args.output_dir / "regime_portrait.png",
                         title="Synthetic kinematic substrate", limits=limits)
    _path, explained = plot_evidence_geometry(
        embeddings, runs, args.output_dir / "evidence_geometry.png",
        title="Synthetic kinematic (13-d state vector)",
    )
    print(f"\nEvidence geometry: 2 PCA components explain {explained:.1%} of variance")

    (args.output_dir / "summary.json").write_text(
        json.dumps({
            "provenance": provenance,
            "headline_threshold_free": separation,
            "fixed_threshold_operating_point": detector,
            "threshold_sweep": sweep,
            "pca_explained_variance_2c": explained,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    keys = list(next(iter(separation.values())).keys())
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario", *keys])
        writer.writeheader()
        for name, metrics in separation.items():
            writer.writerow({"scenario": name, **metrics})

    print(f"\nwrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
