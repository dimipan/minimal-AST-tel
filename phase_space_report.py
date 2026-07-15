"""Render an AST phase portrait from any set of ast-tau-v1.1 trajectories.

This is the phase-space layer's entry point. It imports no substrate and no
binding -- it reads the shared trajectory contract and classifies telemetry. Point
it at trajectories from ANY substrate:

    python phase_space_report.py outputs/boir/*.jsonl
    python phase_space_report.py outputs/kinematic/*.jsonl --out portrait.png

It prints a per-trajectory regime breakdown and the pathology verdict, and writes
the phase portrait.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase_space import (
    CHURN_FRACTION_CUT,
    is_pathological,
    load_phase,
    phase_signature,
    regime_counts,
    REGIME_ORDER,
)
from plotting import plot_phase_portrait


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path, nargs="+",
                        help="one or more ast-tau-v1.1 JSONL files")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--si-threshold", type=float, default=0.20)
    parser.add_argument("--title", default="AST phase space")
    args = parser.parse_args()

    labelled = {}
    summary = {}
    for path in args.trajectory:
        header, points = load_phase(path, si_threshold=args.si_threshold)
        name = f"{header['substrate'].split('-')[-1]}/{header['session_id']}"
        # keep names short and unique
        name = header["session_id"]
        labelled[name] = points
        counts = regime_counts(points)
        sig = phase_signature(points)
        summary[name] = {
            "regimes": counts,
            "pathology_index": sig["pathology_index"],
            "pathological": is_pathological(points),
            "terminal_regime": sig["terminal_regime"],
        }

    width = max(len(n) for n in summary)
    header_row = f"  {'trajectory':{width}s}  " + " ".join(f"{r[:5]:>5s}" for r in REGIME_ORDER)
    header_row += f"  {'fric|press':>10s}  verdict"
    print(header_row)
    print("  " + "-" * (len(header_row) - 2))
    for name, info in summary.items():
        counts = info["regimes"]
        row = f"  {name:{width}s}  " + " ".join(f"{counts[r]:5d}" for r in REGIME_ORDER)
        verdict = "PATHOLOGICAL CHURN" if info["pathological"] else "healthy"
        row += f"  {info['pathology_index']:10.3f}  {verdict}"
        print(row)
    print(f"\n  churn-fraction cut = {CHURN_FRACTION_CUT:.2f} (share of exchanges stuck at zero yield with high friction)")

    out = args.out or (args.trajectory[0].parent / "phase_portrait.png")
    plot_phase_portrait(labelled, out, title=args.title, si_threshold=args.si_threshold)
    print(f"  wrote {out}")

    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {json_out}")


if __name__ == "__main__":
    main()
