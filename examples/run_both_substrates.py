"""Run one compact smoke trajectory from each substrate.

The hiker arm explicitly uses hash embeddings so this example works in CI.
Use exp1_evaluation.py without --hash-embeddings for the Qwen3 experiment.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from binding_kinematic import KINEMATIC_SCENARIOS, make_kinematic_binding
from corpus_hiker import SCENARIOS, make_hiker_binding
from substrate_binding import run_binding


if __name__ == "__main__":
    out = ROOT / "outputs" / "examples"
    hiker = run_binding(
        make_hiker_binding(use_hash_embeddings=True),
        SCENARIOS["agent_repetition"],
        output_path=out / "hiker.jsonl",
        session_id="hiker-example",
    )
    robot = run_binding(
        make_kinematic_binding(),
        KINEMATIC_SCENARIOS["orbit_stall"],
        output_path=out / "kinematic.jsonl",
        session_id="kinematic-example",
    )
    print(f"hiker peak SI: {max(r['si'] for r in hiker):.3f}")
    print(f"kinematic peak SI: {max(r['si'] for r in robot):.3f}")
