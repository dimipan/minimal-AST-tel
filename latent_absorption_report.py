"""Latent-absorption report: the novelty x yield quadrant view.

Reads the shared ast-tau-v1.1 contract, classifies each exchange into one of four
quadrants, prints occupancy and the absorption verdict, runs the paired-decoy
validation, and writes the quadrant scatter. Imports no binding, no monitor.

    python latent_absorption_report.py outputs/boir/*.jsonl
    python latent_absorption_report.py outputs/kinematic/*.jsonl

The absorption quadrant is GEOMETRIC: it sees low novelty and low yield, but it
cannot by itself tell "stuck with work remaining" from "finished and idle". That
distinction needs residual pressure (PE), which is what phase_space.py adds. The
two views are complementary: this one names the KIND of non-progress; phase_space
says whether the schema is still open. Read them together.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from latent_absorption import (
    ABSORPTION_FRACTION_CUT,
    LATENT_ABSORPTION,
    QUADRANTS,
    USEFUL_RECURRENCE,
    absorption_signature,
    is_absorbed,
    load_absorption,
    paired_decoy_check,
)
from plotting import plot_absorption_plane

SHORT = {
    "productive_expansion": "PE",
    "useful_recurrence": "UR",
    "unproductive_expansion": "UE",
    "latent_absorption": "LA",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--title", default="Latent-absorption plane (ON x Y)")
    args = parser.parse_args()

    labelled, summary = {}, {}
    for path in args.trajectory:
        header, points = load_absorption(path)
        name = header["session_id"]
        labelled[name] = points
        sig = absorption_signature(points)
        counts = Counter(p.quadrant for p in points)
        summary[name] = {
            "quadrant_counts": {q: counts[q] for q in QUADRANTS},
            "latent_absorption_fraction": sig["latent_absorption_fraction"],
            "terminal_quadrant": sig["terminal_quadrant"],
            "absorbed": is_absorbed(points),
        }

    width = max(len(n) for n in summary)
    print(f"  {'trajectory':{width}s}   PE   UR   UE   LA   LA_frac  terminal  absorbed")
    print("  " + "-" * (width + 44))
    for name, info in summary.items():
        c = info["quadrant_counts"]
        print(f"  {name:{width}s}  {c['productive_expansion']:3d}  "
              f"{c['useful_recurrence']:3d}  {c['unproductive_expansion']:3d}  "
              f"{c['latent_absorption']:3d}  {info['latent_absorption_fraction']:7.3f}  "
              f"{SHORT[info['terminal_quadrant']]:>8s}  "
              f"{'ABSORBED' if info['absorbed'] else 'healthy'}")
    print(f"\n  absorption-fraction cut = {ABSORPTION_FRACTION_CUT:.2f}")
    print("  note: the absorption quadrant is geometric. A finished-and-idle process")
    print("  also shows low novelty and low yield; use phase_space.py (which adds")
    print("  residual pressure PE) to separate 'stuck' from 'done'.")

    # Paired-decoy validation, if we can find a same-geometry / different-yield pair.
    _paired_decoy_report(labelled)

    out = args.out or (args.trajectory[0].parent / "absorption_plane.png")
    plot_absorption_plane(labelled, out, title=args.title)
    print(f"\n  wrote {out}")
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {out.with_suffix('.json')}")


def _paired_decoy_report(labelled) -> None:
    """Find a low-novelty absorbed run and a low-novelty productive run in the set,
    and show that yield alone flips their quadrant."""
    absorbed = [(n, p) for n, p in labelled.items() if is_absorbed(p)]
    healthy = [(n, p) for n, p in labelled.items() if not is_absorbed(p)]
    if not absorbed or not healthy:
        return

    ab_name, ab_pts = absorbed[0]
    he_name, he_pts = healthy[0]
    ab_low = [p for p in ab_pts if p.on < 0.5]
    he_low = [p for p in he_pts if p.on < 0.5]
    if not ab_low or not he_low:
        return

    result = paired_decoy_check(
        he_low, ab_low,
        quadrant_high=USEFUL_RECURRENCE, quadrant_low=LATENT_ABSORPTION,
    )
    print("\n  Paired-decoy validation (real trajectories, same low-novelty geometry)")
    print(f"    high-yield arm ({he_name}): ON~{result['mean_on_high']:.2f} "
          f"Y~{result['mean_y_high']:.2f} -> {SHORT[result['high_yield_quadrant']]}")
    print(f"    low-yield  arm ({ab_name}): ON~{result['mean_on_low']:.2f} "
          f"Y~{result['mean_y_low']:.2f} -> {SHORT[result['low_yield_quadrant']]}")
    verdict = ("PASS: yield alone flips the quadrant -> the plane uses both axes"
               if result["dissociates"] else
               "FAIL: same quadrant despite different yield -> yield axis ignored")
    print(f"    {verdict}")


if __name__ == "__main__":
    main()
