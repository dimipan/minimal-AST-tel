"""State invariants, telemetry bounds, and binding-contract enforcement."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from binding_kinematic import KINEMATIC_SCENARIOS, make_kinematic_binding  # noqa: E402
from corpus_hiker import SCENARIOS, make_hiker_binding  # noqa: E402
from substrate_binding import (  # noqa: E402
    SubstrateBinding,
    SubstrateKnowledgeState,
    run_binding,
)


def _all_runs():
    hiker = make_hiker_binding(use_hash_embeddings=True)
    kinematic = make_kinematic_binding()
    for name, stream in SCENARIOS.items():
        yield f"hiker/{name}", hiker, run_binding(hiker, stream, session_id=name)
    for name, stream in KINEMATIC_SCENARIOS.items():
        yield f"kinematic/{name}", kinematic, run_binding(kinematic, stream, session_id=name)


class TestStateInvariants(unittest.TestCase):
    def test_upsilon_is_monotone_non_decreasing(self):
        for label, binding, records in _all_runs():
            with self.subTest(run=label):
                for dim in binding.schema:
                    series = [r["upsilon"][dim] for r in records]
                    for previous, current in zip(series, series[1:]):
                        self.assertGreaterEqual(
                            current, previous - 1e-12, f"{label}/{dim} decreased"
                        )

    def test_upsilon_bounded(self):
        for label, binding, records in _all_runs():
            with self.subTest(run=label):
                for record in records:
                    for value in record["upsilon"].values():
                        self.assertGreaterEqual(value, 0.0)
                        self.assertLessEqual(value, 1.0)

    def test_gain_equals_upsilon_difference(self):
        for label, binding, records in _all_runs():
            with self.subTest(run=label):
                previous = {dim: 0.0 for dim in binding.schema}
                for record in records:
                    for dim in binding.schema:
                        expected = record["upsilon"][dim] - previous[dim]
                        self.assertAlmostEqual(record["gains"][dim], expected, places=10)
                    previous = dict(record["upsilon"])

    def test_state_rejects_dimension_change(self):
        state = SubstrateKnowledgeState(["a"], 4)
        state.update(np.array([1.0, 0.0, 0.0, 0.0]), {"a": 0.5}, "a")
        with self.assertRaises(ValueError):
            state.update(np.array([1.0, 0.0]), {"a": 0.6}, "a")


class TestTelemetryBounds(unittest.TestCase):
    def test_si_in_unit_interval(self):
        for label, _binding, records in _all_runs():
            with self.subTest(run=label):
                for record in records:
                    self.assertTrue(np.isfinite(record["si"]))
                    self.assertGreaterEqual(record["si"], 0.0)
                    self.assertLessEqual(record["si"], 1.0)

    def test_pe_is_finite_and_non_negative(self):
        for label, _binding, records in _all_runs():
            with self.subTest(run=label):
                for record in records:
                    for value in record["pe"].values():
                        self.assertTrue(np.isfinite(value))
                        self.assertGreaterEqual(value, 0.0)

    def test_eta_and_rho_are_complementary_and_bounded(self):
        for label, _binding, records in _all_runs():
            with self.subTest(run=label):
                for record in records:
                    for dim, eta in record["eta"].items():
                        self.assertGreaterEqual(eta, 0.0)
                        self.assertLessEqual(eta, 1.0)
                        self.assertAlmostEqual(record["rho"][dim], 1.0 - eta, places=10)

    def test_first_exchange_has_no_prior_subspace(self):
        for label, _binding, records in _all_runs():
            with self.subTest(run=label):
                self.assertEqual(records[0]["si"], 0.0)
                self.assertEqual(records[0]["eta"], {})


class TestBindingContract(unittest.TestCase):
    def _binding(self, **overrides):
        kwargs = dict(
            name="t",
            schema=("a", "b"),
            scorer=lambda e: {"a": 0.5},
            embedder=lambda e: np.array([1.0, 0.0]),
            target_selector=lambda e: "a",
        )
        kwargs.update(overrides)
        return SubstrateBinding(**kwargs)

    def test_duplicate_schema_rejected(self):
        with self.assertRaises(ValueError):
            self._binding(schema=("a", "a"))

    def test_importance_outside_schema_rejected(self):
        with self.assertRaises(ValueError):
            self._binding(importance={"z": 1.0})

    def test_score_outside_schema_rejected(self):
        with self.assertRaises(ValueError):
            self._binding(scorer=lambda e: {"z": 0.5}).score(object())

    def test_score_outside_unit_interval_rejected(self):
        with self.assertRaises(ValueError):
            self._binding(scorer=lambda e: {"a": 1.5}).score(object())

    def test_target_outside_schema_rejected(self):
        with self.assertRaises(ValueError):
            self._binding(target_selector=lambda e: "z").target(object())

    def test_zero_norm_embedding_rejected(self):
        with self.assertRaises(ValueError):
            self._binding(embedder=lambda e: np.zeros(2)).embed(object())

    def test_embed_returns_unit_norm(self):
        vector = self._binding(embedder=lambda e: np.array([3.0, 4.0])).embed(object())
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
