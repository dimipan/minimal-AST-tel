"""Check that real qwen3 embeddings preserve the regime separation.

Run this once, on the machine with Ollama, immediately after
``python precompute_embeddings.py``. It is the gate between "the plumbing works
on hash vectors" and "the hiker experiment is reportable".

Why this exists
---------------
Transformer embedding spaces are anisotropic: cosine similarities between
unrelated texts cluster high rather than near zero. SI^perp is built from
orthogonal novelty, so a compressed embedding space lifts SI on EVERY exchange,
including the efficient control. The hash-vector smoke run cannot detect this,
because hash vectors are near-orthogonal by construction.

    python check_embedding_separation.py

Exit code 0 means the qwen3 archive reproduces the ordering the equations
predict. A non-zero exit is not a bug in AST; it means the fixed threshold is
the wrong readout for this embedding space, and mean-SI separation should be
reported instead. Read the printed diagnosis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from corpus_hiker import SCENARIOS, iter_all_evidence, make_hiker_binding
from substrate_binding import run_binding

THRESHOLD = 0.20


def _mean_si(records) -> float:
    return float(np.mean([r["si"] for r in records]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=Path("data/hiker_embeddings_qwen3.npz"),
    )
    parser.add_argument("--si-threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    binding = make_hiker_binding(embeddings_path=args.embeddings_path)

    # 1. Anisotropy of the embedding space itself.
    vectors = np.stack([binding.embed(e) for e in iter_all_evidence()])
    gram = vectors @ vectors.T
    off_diagonal = gram[~np.eye(len(vectors), dtype=bool)]
    print("Embedding space")
    print(f"  vectors           : {vectors.shape[0]} x {vectors.shape[1]}")
    print(f"  pairwise cosine   : p05={np.percentile(off_diagonal, 5):.3f} "
          f"median={np.median(off_diagonal):.3f} "
          f"p95={np.percentile(off_diagonal, 95):.3f}")
    print(f"  spread (p95-p05)  : {np.percentile(off_diagonal, 95) - np.percentile(off_diagonal, 5):.3f}")
    if np.median(off_diagonal) > 0.90:
        print("  NOTE: strongly anisotropic. Expect SI to be lifted on every exchange.")

    # 2. Regime separation.
    runs = {name: run_binding(binding, stream, session_id=name)
            for name, stream in SCENARIOS.items()}
    print("\nRegime separation (qwen3-embedding)")
    for name, records in runs.items():
        si = [r["si"] for r in records]
        print(f"  {name:26s} mean={np.mean(si):.3f} max={np.max(si):.3f} "
              f"n_over_theta={sum(s >= args.si_threshold for s in si)}/{len(si)}")

    control = runs["efficient"]
    control_mean = _mean_si(control)
    control_max = max(r["si"] for r in control)
    stalls = ("agent_repetition", "interviewee_degradation")

    ordering_ok = all(_mean_si(runs[n]) > control_mean for n in stalls)
    threshold_ok = control_max < args.si_threshold and all(
        max(r["si"] for r in runs[n] if r["ground_truth"]["stall"]) >= args.si_threshold
        for n in stalls
    )

    print("\nVerdict")
    print(f"  ordering (stall > control)          : {'PASS' if ordering_ok else 'FAIL'}")
    print(f"  fixed threshold theta={args.si_threshold:.2f} usable : "
          f"{'PASS' if threshold_ok else 'FAIL'}")

    if ordering_ok and threshold_ok:
        print("\n  Report the hiker experiment as-is.")
        return 0
    if ordering_ok and not threshold_ok:
        print(
            "\n  The signal is present but the fixed threshold does not separate the\n"
            "  regimes in this embedding space. This is an honest, reportable outcome:\n"
            "  report MEAN-SI separation across regimes and drop the fixed-threshold\n"
            "  detector table for 9-exchange sessions. Do not retune theta to rescue it."
        )
        return 1
    print(
        "\n  The ordering itself failed. Do not report the hiker experiment. Inspect the\n"
        "  embedding archive (wrong model? unnormalised? wrong text field?) before\n"
        "  drawing any conclusion about AST."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
