"""Regime separation, and the claim that both substrates emit one schema.

The regime tests are ORDINAL, not absolute. They assert that stalls rank above
controls, which is the behaviour the equations promise. They deliberately do not
pin exact SI values: those depend on the embedding map, and the hiker arm here
runs on hash vectors, which are not the experiment.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from binding_kinematic import KINEMATIC_SCENARIOS, make_kinematic_binding  # noqa: E402
from corpus_hiker import SCENARIOS, make_hiker_binding  # noqa: E402
from substrate_binding import TrajectoryWriter, load_trajectory, run_binding  # noqa: E402

REQUIRED_KEYS = {
    "record", "t", "y_ref", "i_t", "upsilon", "delta_upsilon",
    "PE", "PE_total", "knowledge_mean", "gain_total", "SI",
    "eta", "rho", "dampening", "gt",
}


def _mean_si(records):
    return sum(r["si"] for r in records) / len(records)


class TestLinguisticRegimes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binding = make_hiker_binding(use_hash_embeddings=True)
        cls.runs = {
            name: run_binding(binding, stream, session_id=name)
            for name, stream in SCENARIOS.items()
        }

    def test_stall_regimes_exceed_efficient_control(self):
        control = _mean_si(self.runs["efficient"])
        for name in ("agent_repetition", "interviewee_degradation"):
            with self.subTest(regime=name):
                self.assertGreater(_mean_si(self.runs[name]), control)

    def test_efficient_control_stays_below_threshold(self):
        threshold = 0.20
        for record in self.runs["efficient"]:
            self.assertLess(record["si"], threshold)

    def test_labelled_stalls_score_above_unlabelled(self):
        for name in ("agent_repetition", "interviewee_degradation"):
            with self.subTest(regime=name):
                records = self.runs[name]
                stall = [r["si"] for r in records if r["ground_truth"]["stall"]]
                clean = [r["si"] for r in records if not r["ground_truth"]["stall"]]
                self.assertTrue(stall and clean)
                self.assertGreater(min(stall), max(clean))

    def test_paraphrastic_stall_outranks_exact_repetition(self):
        # Repeating an already-resolved category (location, upsilon 0.78) is damped
        # by the (1 - upsilon) saturation term; hammering an unresolved one
        # (equipment, upsilon 0.20) is not. The equations should show that.
        self.assertGreater(
            _mean_si(self.runs["interviewee_degradation"]),
            _mean_si(self.runs["agent_repetition"]),
        )


class TestKinematicRegimes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binding = make_kinematic_binding()
        cls.runs = {
            name: run_binding(binding, stream, session_id=name)
            for name, stream in KINEMATIC_SCENARIOS.items()
        }

    def test_orbit_stall_is_detected(self):
        self.assertGreater(_mean_si(self.runs["orbit_stall"]), _mean_si(self.runs["clean"]))
        stall = [r["si"] for r in self.runs["orbit_stall"] if r["ground_truth"]["stall"]]
        self.assertGreater(sum(stall) / len(stall), 0.20)

    def test_saturated_hold_is_suppressed(self):
        # The I2 control: recurrence inside a fully resolved schema must NOT read as
        # a stall, because (1 - upsilon) collapses the contribution. This is the
        # single most important negative control in the repository.
        hold = self.runs["saturated_hold"]
        tail = hold[-8:]
        self.assertTrue(all(r["si"] < 0.20 for r in tail))
        self.assertLessEqual(_mean_si(hold), _mean_si(self.runs["clean"]) + 0.02)

    def test_orbit_outranks_hold(self):
        self.assertGreater(
            _mean_si(self.runs["orbit_stall"]), _mean_si(self.runs["saturated_hold"])
        )


class TestSharedTrajectoryContract(unittest.TestCase):
    """Phase space must be able to read both substrates with no branching."""

    def test_both_substrates_emit_identical_record_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run_binding(
                make_hiker_binding(use_hash_embeddings=True),
                SCENARIOS["interviewee_degradation"],
                output_path=tmp / "hiker.jsonl",
                session_id="hiker",
            )
            run_binding(
                make_kinematic_binding(),
                KINEMATIC_SCENARIOS["orbit_stall"],
                output_path=tmp / "kinematic.jsonl",
                session_id="kinematic",
            )

            shapes = []
            for name in ("hiker.jsonl", "kinematic.jsonl"):
                header, exchanges = load_trajectory(tmp / name)
                self.assertEqual(header["schema_version"], TrajectoryWriter.SCHEMA_VERSION)
                self.assertIn("task_schema_M", header)
                self.assertTrue(exchanges)
                for row in exchanges:
                    self.assertEqual(set(row) - REQUIRED_KEYS, set())
                    self.assertEqual(REQUIRED_KEYS - set(row), set())
                shapes.append({k: type(exchanges[0][k]).__name__ for k in REQUIRED_KEYS})
            self.assertEqual(shapes[0], shapes[1])

    def test_eta_is_present_for_phase_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "k.jsonl"
            run_binding(
                make_kinematic_binding(),
                KINEMATIC_SCENARIOS["orbit_stall"],
                output_path=path,
                session_id="k",
            )
            _header, exchanges = load_trajectory(path)
            # novelty (eta) against yield (delta_upsilon) is the phase-space plane;
            # both must be readable from the file alone.
            with_eta = [row for row in exchanges if row["eta"]]
            self.assertTrue(with_eta)
            for row in with_eta:
                self.assertTrue(all(0.0 <= v <= 1.0 for v in row["eta"].values()))
                self.assertIsNotNone(row["dampening"])

    def test_jsonl_is_plain_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "k.jsonl"
            run_binding(
                make_kinematic_binding(),
                KINEMATIC_SCENARIOS["clean"],
                output_path=path,
                session_id="k",
            )
            for line in path.read_text().splitlines():
                json.loads(line)


if __name__ == "__main__":
    unittest.main()
