"""Evaluate AST on the fully synthetic SAR hiker substrate.

Reporting policy
----------------
The HEADLINE metrics are threshold-free (AUC, mean-SI separation). This is not a
presentational choice; it follows from a measured property of the monitor.

SI^perp's ORDERING is a property of the monitor. Its ABSOLUTE SCALE is a property
of the evidence representation: the normaliser sum_i rho_i grows when the
embedding places non-trivial membership mass on non-target dimensions, which
compresses SI without weakening rank separation. The identical stall exchanges
score mean SI 0.217 under 96-d hash vectors and 0.174 under 4096-d
qwen3-embedding, while AUC is 1.000 in both.

A threshold fitted under one embedding map therefore does not transfer to
another, and with 9-exchange sessions the false-positive rate resolves only to
1/9 -- too coarse to fit theta honestly. The fixed-threshold table is still
printed, at the operating point declared in the manifest, so that the failure is
visible rather than hidden. Do not retune theta to make it look better.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from corpus_hiker import HIKER_DIMENSIONS, SCENARIOS, make_hiker_binding
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


def _provenance(binding, args) -> dict:
    """Say out loud which representation actually ran.

    No silent fallback exists -- a missing archive raises, and hash vectors need
    an explicit flag -- but this banner means nobody has to take that on trust.
    """
    vectors = np.stack([binding.embed(e) for e in SCENARIOS["efficient"]])
    gram = vectors @ vectors.T
    off = gram[~np.eye(len(vectors), dtype=bool)]

    manifest_path = args.embeddings_path.with_suffix(".manifest.json")
    archive = {}
    if not args.hash_embeddings and manifest_path.exists():
        archive = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "binding": binding.name,
        "embedding_backend": "hash-smoke" if args.hash_embeddings else "qwen3-embedding",
        "embedding_dimension": int(vectors.shape[1]),
        "embedding_model": archive.get("model"),
        "embedding_archive": None if args.hash_embeddings else str(args.embeddings_path),
        "archive_sha256": archive.get("archive_sha256"),
        "texts_sha256": archive.get("texts_sha256"),
        "median_pairwise_cosine": round(float(np.median(off)), 4),
        "cosine_spread_p95_p05": round(
            float(np.percentile(off, 95) - np.percentile(off, 5)), 4
        ),
        "calibration": MANIFEST["manifest_version"],
        "declared_threshold": MANIFEST["per_binding_calibration"]
        .get(binding.name, {})
        .get("offline_detection_threshold"),
    }


def _fmt(value, width=7):
    return f"{'--':>{width}}" if value is None else f"{value:{width}.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS),
                        default=list(SCENARIOS))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hiker"))
    parser.add_argument("--embeddings-path", type=Path,
                        default=Path("data/hiker_embeddings_qwen3.npz"),
                        help="precomputed qwen3-embedding NPZ archive")
    parser.add_argument("--hash-embeddings", action="store_true",
                        help="deterministic hash vectors; smoke testing only")
    parser.add_argument("--si-threshold", type=float, default=0.20,
                        help="reported operating point; NOT fitted on this substrate")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    binding = make_hiker_binding(
        embeddings_path=args.embeddings_path,
        use_hash_embeddings=args.hash_embeddings,
    )

    provenance = _provenance(binding, args)
    print("\nRepresentation provenance")
    for key, value in provenance.items():
        print(f"  {key:26s} {value}")
    if provenance["embedding_backend"] == "hash-smoke":
        print("\n  WARNING: hash vectors. This is the smoke path, not the experiment.")

    runs, embeddings = {}, {}
    for scenario in args.scenarios:
        runs[scenario] = run_binding(
            binding,
            SCENARIOS[scenario],
            output_path=args.output_dir / f"{scenario}.jsonl",
            session_id=scenario,
            meta={
                "synthetic_corpus": True,
                "modality": "linguistic",
                **{k: v for k, v in provenance.items() if v is not None},
            },
        )
        embeddings[scenario] = np.stack([binding.embed(e) for e in SCENARIOS[scenario]])

    limits = substrate_limits(runs, si_threshold=args.si_threshold)

    # -- headline: threshold-free -------------------------------------------
    separation = {name: separation_metrics(records) for name, records in runs.items()}
    print("\nHeadline (threshold-free)")
    print(f"  {'condition':26s} {'AUC':>7s} {'meanSI stall':>13s} "
          f"{'meanSI clean':>13s} {'separation':>11s} {'rank-perfect':>13s}")
    for name, m in separation.items():
        print(f"  {name:26s} {_fmt(m['auc'])} {_fmt(m['mean_si_stall'], 13)} "
              f"{_fmt(m['mean_si_clean'], 13)} {_fmt(m['si_separation'], 11)} "
              f"{str(m['perfect_rank_separation']):>13s}")

    # -- reported operating point, not a fitted one --------------------------
    detector = {
        name: detector_metrics(records, threshold=args.si_threshold)
        for name, records in runs.items()
    }
    sweep = threshold_sweep(runs)
    print(f"\nFixed-threshold operating point (theta = {args.si_threshold:.2f}; "
          f"declared for this binding: {provenance['declared_threshold']})")
    for name, m in detector.items():
        print(f"  {name:26s} recall={m['recall']} precision={m['precision']} "
              f"fpr={m['false_positive_rate']}")
    print("\nPooled threshold sweep (reported so that no operating point has to be chosen)")
    print(format_sweep(sweep))

    # -- figures -------------------------------------------------------------
    for name, records in runs.items():
        label = f"Synthetic SAR — {name.replace('_', ' ')}"
        plot_trajectory(records, HIKER_DIMENSIONS, args.output_dir / f"{name}.png",
                        title=label, limits=limits)
        plot_telemetry_trajectory(
            records, args.output_dir / f"{name}_telemetry_trajectory.png",
            title=label, limits=limits,
        )
    plot_scenario_comparison(runs, args.output_dir / "scenario_comparison.png",
                             title="Synthetic SAR substrate", limits=limits)
    plot_regime_portrait(runs, args.output_dir / "regime_portrait.png",
                         title="Synthetic SAR substrate", limits=limits)
    _path, explained = plot_evidence_geometry(
        embeddings, runs, args.output_dir / "evidence_geometry.png",
        title=f"Synthetic SAR ({provenance['embedding_backend']})",
    )
    print(f"\nEvidence geometry: 2 PCA components explain {explained:.1%} of variance")

    # -- artefacts -----------------------------------------------------------
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
