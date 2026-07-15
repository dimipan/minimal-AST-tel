"""Tests for the latent-absorption (novelty x yield) view.

Pins three things: the quadrant algebra, that the plane needs BOTH axes (the
paired-decoy control ported from the benchmark, driven by real trajectories), and
that latent absorption agrees with phase_space's churn on the genuine stalls while
honestly diverging on the finished-and-idle case.
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from binding_boir import run_boir_ast  # noqa: E402
from binding_kinematic import KINEMATIC_SCENARIOS, make_kinematic_binding  # noqa: E402
from latent_absorption import (  # noqa: E402
    LATENT_ABSORPTION,
    PRODUCTIVE_EXPANSION,
    UNPRODUCTIVE_EXPANSION,
    USEFUL_RECURRENCE,
    absorption_signature,
    is_absorbed,
    load_absorption,
    normalised_yield,
    orthogonal_novelty,
    paired_decoy_check,
    quadrant_scores,
    trajectory_to_absorption,
)
from phase_space import is_pathological, trajectory_to_phase  # noqa: E402
from substrate_binding import run_binding  # noqa: E402
from synthetic_intent_trajectories import all_scenarios  # noqa: E402


def _boir():
    out = {}
    for name, traj in all_scenarios().items():
        for epoch, records in run_boir_ast(traj).items():
            key = f"{name}/e{epoch}" if len(traj.epoch_boundaries) > 1 else name
            out[key] = records
    return out


class TestQuadrantAlgebra(unittest.TestCase):
    def test_corners(self):
        self.assertEqual(max(quadrant_scores(1.0, 1.0), key=quadrant_scores(1.0, 1.0).get),
                         PRODUCTIVE_EXPANSION)
        self.assertEqual(max(quadrant_scores(0.0, 1.0), key=quadrant_scores(0.0, 1.0).get),
                         USEFUL_RECURRENCE)
        self.assertEqual(max(quadrant_scores(1.0, 0.0), key=quadrant_scores(1.0, 0.0).get),
                         UNPRODUCTIVE_EXPANSION)
        self.assertEqual(max(quadrant_scores(0.0, 0.0), key=quadrant_scores(0.0, 0.0).get),
                         LATENT_ABSORPTION)

    def test_scores_are_a_partition_of_unity(self):
        for on in (0.0, 0.3, 0.7, 1.0):
            for y in (0.0, 0.4, 1.0):
                self.assertAlmostEqual(sum(quadrant_scores(on, y).values()), 1.0, places=12)

    def test_novelty_defaults_to_max_without_prior_subspace(self):
        # An exchange with no contributing eta has no established subspace to fall
        # inside, so it is maximally novel by convention.
        self.assertEqual(orthogonal_novelty({"eta": {}}), 1.0)

    def test_yield_uses_absolute_floor_not_peak(self):
        # A run of small-but-real gains must read as high yield throughout; a
        # peak-relative scale would wrongly demote all but the largest.
        records = [{"gain_total": g} for g in (0.8, 0.2, 0.5, 0.1)]
        ys = normalised_yield(records)
        self.assertTrue(all(y >= 0.99 for y in ys), "modest real gains demoted to low yield")

    def test_zero_gain_is_zero_yield(self):
        records = [{"gain_total": 0.0} for _ in range(5)]
        self.assertTrue(all(y == 0.0 for y in normalised_yield(records)))


class TestPairedDecoy(unittest.TestCase):
    """The plane must use BOTH axes: same geometry, different yield -> different
    quadrant. Ported control from the latent-absorption benchmark, real telemetry."""

    def test_same_low_novelty_geometry_yield_flips_quadrant(self):
        runs = _boir()
        clean = [p for p in trajectory_to_absorption(runs["clean_convergence"]) if p.on < 0.5]
        stall = [p for p in trajectory_to_absorption(runs["ambiguous_recurrence"]) if p.on < 0.5]
        self.assertTrue(clean and stall)
        result = paired_decoy_check(
            clean, stall,
            quadrant_high=USEFUL_RECURRENCE, quadrant_low=LATENT_ABSORPTION,
        )
        self.assertTrue(result["dissociates"], "yield axis ignored: same quadrant")
        self.assertTrue(result["matches_expected"])
        # geometry really is comparable; only yield differs
        self.assertLess(abs(result["mean_on_high"] - result["mean_on_low"]), 0.25)
        self.assertGreater(result["mean_y_high"] - result["mean_y_low"], 0.5)


class TestAbsorptionSeparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = _boir()

    def test_only_ambiguous_recurrence_is_absorbed(self):
        for name, records in self.runs.items():
            points = trajectory_to_absorption(records)
            self.assertEqual(
                is_absorbed(points), name.startswith("ambiguous_recurrence"),
                f"{name}: absorption verdict wrong",
            )

    def test_healthy_boir_runs_are_useful_recurrence(self):
        # BOIR keeps refining the same goal posteriors: low novelty, real yield.
        for name, records in self.runs.items():
            if name.startswith("ambiguous_recurrence"):
                continue
            sig = absorption_signature(trajectory_to_absorption(records))
            self.assertEqual(sig["terminal_quadrant"], USEFUL_RECURRENCE, name)

    def test_stall_terminates_in_absorption(self):
        sig = absorption_signature(trajectory_to_absorption(self.runs["ambiguous_recurrence"]))
        self.assertEqual(sig["terminal_quadrant"], LATENT_ABSORPTION)


class TestSubstrateAgnostic(unittest.TestCase):
    def test_reads_kinematic_with_no_binding_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "orbit.jsonl"
            run_binding(make_kinematic_binding(), KINEMATIC_SCENARIOS["orbit_stall"],
                        output_path=path, session_id="orbit")
            header, points = load_absorption(path)
            self.assertEqual(header["schema_version"], "ast-tau-v1.1")
            self.assertTrue(is_absorbed(points))

    def test_agreement_and_honest_divergence_with_phase_space(self):
        """On genuine stalls the two views agree. On the finished-and-idle robot
        they diverge, and that divergence is the point: absorption is geometric and
        cannot see that the schema is closed; phase_space (with PE) can."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            verdicts = {}
            for scenario in ("clean", "orbit_stall", "saturated_hold"):
                path = tmp / f"{scenario}.jsonl"
                run_binding(make_kinematic_binding(), KINEMATIC_SCENARIOS[scenario],
                            output_path=path, session_id=scenario)
                _h, abs_pts = load_absorption(path)
                from phase_space import load_phase
                _h2, phase_pts = load_phase(path)
                verdicts[scenario] = (is_absorbed(abs_pts), is_pathological(phase_pts))

            # genuine stall: both fire
            self.assertEqual(verdicts["orbit_stall"], (True, True))
            # clean: neither fires
            self.assertEqual(verdicts["clean"], (False, False))
            # finished-and-idle: absorption may fire (geometric), phase_space does
            # NOT (PE has drained -> converged, not churn). This is the documented
            # divergence, not a bug.
            self.assertFalse(verdicts["saturated_hold"][1],
                             "phase_space must call a finished hold converged, not churn")


if __name__ == "__main__":
    unittest.main()
