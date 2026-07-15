"""Tests for the phase-space layer.

The phase space is a reader over the shared trajectory contract. These tests pin
two things: that it stays substrate-agnostic (reads any binding's output with no
binding import), and that it puts the pathological stall -- and only it -- in the
churn corner.
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from binding_boir import run_boir_ast  # noqa: E402
from binding_kinematic import KINEMATIC_SCENARIOS, make_kinematic_binding  # noqa: E402
from phase_space import (
    CHURN_FRACTION_CUT,  # noqa: E402
    CHURN,
    CONVERGED,
    PRODUCTIVE,
    QUIET_INCOMPLETE,
    PhaseThresholds,
    classify_exchange,
    is_pathological,
    load_phase,
    phase_signature,
    trajectory_to_phase,
)
from substrate_binding import run_binding  # noqa: E402
from synthetic_intent_trajectories import all_scenarios  # noqa: E402


def _boir_records():
    out = {}
    for name, traj in all_scenarios().items():
        for epoch, records in run_boir_ast(traj).items():
            key = f"{name}/e{epoch}" if len(traj.epoch_boundaries) > 1 else name
            out[key] = records
    return out


class TestClassification(unittest.TestCase):
    def setUp(self):
        self.th = PhaseThresholds(yield_=0.02, si=0.20, pressure=1.0)

    def test_productive_is_any_positive_yield(self):
        r = {"gain_total": 0.5, "SI": 0.9, "PE_total": 2.0}
        self.assertEqual(classify_exchange(r, self.th), PRODUCTIVE)

    def test_churn_is_zero_yield_high_friction(self):
        r = {"gain_total": 0.0, "SI": 0.5, "PE_total": 2.0}
        self.assertEqual(classify_exchange(r, self.th), CHURN)

    def test_converged_is_zero_yield_low_friction_low_pressure(self):
        r = {"gain_total": 0.0, "SI": 0.02, "PE_total": 0.2}
        self.assertEqual(classify_exchange(r, self.th), CONVERGED)

    def test_quiet_incomplete_is_zero_yield_low_friction_high_pressure(self):
        r = {"gain_total": 0.0, "SI": 0.02, "PE_total": 2.0}
        self.assertEqual(classify_exchange(r, self.th), QUIET_INCOMPLETE)

    def test_churn_and_converged_share_zero_yield_and_low_or_high(self):
        # The point of the plane: same yield, separated only by friction+pressure.
        churn = {"gain_total": 0.0, "SI": 0.5, "PE_total": 2.0}
        converged = {"gain_total": 0.0, "SI": 0.02, "PE_total": 0.2}
        self.assertNotEqual(
            classify_exchange(churn, self.th), classify_exchange(converged, self.th)
        )


class TestPathologySeparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = _boir_records()

    def test_only_ambiguous_recurrence_is_pathological(self):
        for name, records in self.runs.items():
            points = trajectory_to_phase(records)
            expected = name.startswith("ambiguous_recurrence")
            self.assertEqual(
                is_pathological(points), expected,
                f"{name}: pathology verdict wrong",
            )

    def test_churn_run_has_a_pathology_gap_over_every_healthy_run(self):
        churn = phase_signature(
            trajectory_to_phase(self.runs["ambiguous_recurrence"])
        )["pathology_index"]
        for name, records in self.runs.items():
            if name.startswith("ambiguous_recurrence"):
                continue
            healthy = phase_signature(trajectory_to_phase(records))["pathology_index"]
            self.assertGreater(churn, 2.0 * healthy, f"gap too small vs {name}")

    def test_confidently_wrong_is_not_churn(self):
        # It resolves cleanly (wrongly). It must read as productive/converged, not
        # as a stall -- the phase space must not mistake a wrong answer for a stall.
        points = trajectory_to_phase(self.runs["confidently_wrong"])
        self.assertFalse(is_pathological(points))
        self.assertEqual(
            sum(p.regime == CHURN for p in points), 0
        )

    def test_churn_run_terminates_in_churn(self):
        points = trajectory_to_phase(self.runs["ambiguous_recurrence"])
        self.assertEqual(phase_signature(points)["terminal_regime"], CHURN)


class TestSubstrateAgnostic(unittest.TestCase):
    def test_reads_a_kinematic_trajectory_with_no_binding_import(self):
        # The layer never imports binding_kinematic; it only reads the JSONL.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "orbit.jsonl"
            run_binding(
                make_kinematic_binding(),
                KINEMATIC_SCENARIOS["orbit_stall"],
                output_path=path,
                session_id="orbit",
            )
            header, points = load_phase(path)
            self.assertEqual(header["schema_version"], "ast-tau-v1.1")
            self.assertTrue(points)
            # the orbit stall should place exchanges in the churn corner
            self.assertGreater(sum(p.regime == CHURN for p in points), 0)

    def test_kinematic_orbit_is_pathological_and_clean_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            verdicts = {}
            for scenario in ("clean", "orbit_stall"):
                path = tmp / f"{scenario}.jsonl"
                run_binding(
                    make_kinematic_binding(),
                    KINEMATIC_SCENARIOS[scenario],
                    output_path=path,
                    session_id=scenario,
                )
                _h, points = load_phase(path)
                verdicts[scenario] = is_pathological(points)
            self.assertTrue(verdicts["orbit_stall"])
            self.assertFalse(verdicts["clean"])


class TestPhaseThresholds(unittest.TestCase):
    def test_pressure_cut_is_scale_free(self):
        records = _boir_records()["ambiguous_recurrence"]
        th = PhaseThresholds.from_trajectory(records)
        pe = [r["pe_total"] for r in records]
        self.assertAlmostEqual(th.pressure, (min(pe) + max(pe)) / 2.0, places=9)

    def test_si_cut_matches_declared_threshold(self):
        th = PhaseThresholds.from_trajectory(
            _boir_records()["clean_convergence"], si_threshold=0.20
        )
        self.assertEqual(th.si, 0.20)


if __name__ == "__main__":
    unittest.main()
