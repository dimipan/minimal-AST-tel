"""Render the global AST telemetry trajectory from any ast-tau-v1.1 JSONL file.

Substrate-independent: it reads the shared contract and never imports a binding.
Named "telemetry trajectory", not "phase space" -- the axes are telemetry
channels, not a state and its derivative. The phase-space package will define
its own portrait over eta (novelty) and delta_upsilon (yield).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plotting import plot_telemetry_trajectory
from substrate_binding import load_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path, nargs="+",
                        help="one or more ast-tau-v1.1 JSONL files")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--si-threshold", type=float, default=0.20)
    args = parser.parse_args()

    loaded = {}
    for path in args.trajectory:
        header, exchanges = load_trajectory(path)
        loaded[path] = (header, [
            {
                "t": row["t"],
                "knowledge_mean": row["knowledge_mean"],
                "si": row["SI"],
                "pe_total": row["PE_total"],
                "gains": row["delta_upsilon"] or {"_": 0.0},
                "upsilon": row["upsilon"],
                "ground_truth": row["gt"],
            }
            for row in exchanges
        ])

    # Shared limits across every file passed in, so the views are comparable.
    runs = {str(path): records for path, (_h, records) in loaded.items()}
    from plotting import substrate_limits
    limits = substrate_limits(runs, si_threshold=args.si_threshold)

    for path, (header, records) in loaded.items():
        out_dir = args.out_dir or path.parent
        output = out_dir / f"{path.stem}_telemetry_trajectory.png"
        plot_telemetry_trajectory(
            records, output,
            title=f"{header['substrate']} — {header['session_id']}",
            limits=limits,
        )
        print(output)


if __name__ == "__main__":
    main()
