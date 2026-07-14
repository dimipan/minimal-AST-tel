"""The refactor must not move a single number.

compute_geometric_subspace_SI was rewritten as a thin wrapper over
compute_geometric_subspace_SI_diagnostic so that eta/rho could be exported.
These tests pin the result against reference_si.py, which is the pre-refactor
implementation vendored verbatim.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from reference_si import reference_si  # noqa: E402
from telemetry_tools_geometric import GeometricTelemetryTool  # noqa: E402
from substrate_binding import SubstrateKnowledgeState  # noqa: E402
from binding_kinematic import KINEMATIC_SCENARIOS, make_kinematic_binding  # noqa: E402
from corpus_hiker import SCENARIOS, make_hiker_binding  # noqa: E402

MANIFEST = json.loads((ROOT / "calibration_manifest.json").read_text())
SI_CFG = MANIFEST["si_geometric_subspace"]
SI_KW = {
    "lambda_damp": float(SI_CFG["lambda_damp"]),
    "eta_min": float(SI_CFG["eta_min"]),
    "membership_threshold": float(SI_CFG["membership_threshold"]),
    "window": SI_CFG["window"],
}


def _replay(binding, evidence_stream):
    """Drive the state by hand so both SI implementations see identical inputs."""
    stream = list(evidence_stream)
    dim = binding.embed(stream[0]).size
    state = SubstrateKnowledgeState(binding.schema, dim)
    history = []
    rows = []
    for evidence in stream:
        embedding = binding.embed(evidence)
        state.update(embedding, binding.score(evidence), binding.target(evidence))
        history.append({"target_dimension": binding.target(evidence)})
        rows.append(
            (
                GeometricTelemetryTool.compute_geometric_subspace_SI(state, history, **SI_KW),
                reference_si(state, list(history), **SI_KW),
                GeometricTelemetryTool.compute_geometric_subspace_SI_diagnostic(
                    state, history, **SI_KW
                ),
            )
        )
    return rows


class TestFrozenCoreRegression(unittest.TestCase):
    def _check(self, binding, scenarios):
        for name, stream in scenarios.items():
            with self.subTest(scenario=name):
                rows = _replay(binding, stream)
                for t, (patched, reference, diagnostic) in enumerate(rows, start=1):
                    self.assertAlmostEqual(
                        patched, reference, places=12,
                        msg=f"{name} t={t}: refactor moved SI",
                    )
                    self.assertAlmostEqual(
                        diagnostic["si"], reference, places=12,
                        msg=f"{name} t={t}: diagnostic disagrees with reference",
                    )

    def test_hiker_si_matches_reference(self):
        self._check(make_hiker_binding(use_hash_embeddings=True), SCENARIOS)

    def test_kinematic_si_matches_reference(self):
        self._check(make_kinematic_binding(), KINEMATIC_SCENARIOS)


class TestDampeningMode(unittest.TestCase):
    """Aggregate-gain dampening is a declared contract, not an accident."""

    def test_manifest_declares_aggregate(self):
        self.assertEqual(SI_CFG["dampening_gain"], "aggregate_delta_upsilon")

    def test_state_exposes_aggregate_gain(self):
        state = SubstrateKnowledgeState(["a", "b", "c"], 4)
        vector = np.array([1.0, 0.0, 0.0, 0.0])
        gains = state.update(vector, {"a": 0.4, "b": 0.3}, "a")
        self.assertAlmostEqual(gains["a"], 0.4)
        self.assertAlmostEqual(gains["b"], 0.3)
        # This is the whole point: the dampening term must see 0.7, not 0.4.
        self.assertAlmostEqual(state.last_total_gain, 0.7)

    def test_dampening_uses_aggregate_not_target_gain(self):
        binding = make_kinematic_binding()
        state = SubstrateKnowledgeState(binding.schema, 13)
        history = []
        multi = None
        for evidence in KINEMATIC_SCENARIOS["clean"]:
            gains = state.update(
                binding.embed(evidence), binding.score(evidence), binding.target(evidence)
            )
            history.append({"target_dimension": binding.target(evidence)})
            diagnostic = GeometricTelemetryTool.compute_geometric_subspace_SI_diagnostic(
                state, history, **SI_KW
            )
            positive = [g for g in gains.values() if g > 0.0]
            if len(positive) >= 2 and diagnostic["eta"]:
                multi = (gains, diagnostic)
        self.assertIsNotNone(
            multi, "clean kinematic run should contain a multi-dimension gain exchange"
        )
        gains, diagnostic = multi
        total = sum(gains.values())
        target_only = max(gains.values())
        expected = max(SI_KW["eta_min"], 1.0 - SI_KW["lambda_damp"] * total)
        self.assertAlmostEqual(diagnostic["delta_upsilon_total"], total, places=12)
        self.assertAlmostEqual(diagnostic["dampening"], expected, places=12)
        self.assertGreater(total, target_only)  # the two modes really do differ here


if __name__ == "__main__":
    unittest.main()
