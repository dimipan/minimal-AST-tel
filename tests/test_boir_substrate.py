"""Tests for the BOIR intent-estimator substrate.

Two of these are not routine assertions. `TestMonotoneIntegrationBoundary` pins
the structural finding that decided the scorer choice, and
`TestConfidentlyWrongControl` pins the limitation the substrate exists to expose.
If either ever flips, the README is wrong.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bayesian_intent import (  # noqa: E402
    RecursiveBayesianIntentEstimator,
    binary_entropy,
    kl_divergence,
)
from binding_boir import (  # noqa: E402
    build_windows,
    embed_inference_window,
    make_boir_binding,
    run_boir_ast,
)
from evaluation_utils import separation_metrics  # noqa: E402
from synthetic_intent_trajectories import (  # noqa: E402
    GOAL_NAMES,
    all_scenarios,
    make_ambiguous_recurrence,
    make_clean_convergence,
    make_confidently_wrong,
    make_intent_switch,
)


class TestEstimator(unittest.TestCase):
    def test_posterior_is_a_distribution(self):
        traj = make_clean_convergence()
        est = RecursiveBayesianIntentEstimator(n_goals=traj.n_goals)
        for p in est.run(traj.angles, traj.paths):
            self.assertAlmostEqual(float(p.sum()), 1.0, places=10)
            self.assertTrue(np.all(p >= 0.0))
            self.assertTrue(np.all(np.isfinite(p)))

    def test_uniform_prior_on_reset(self):
        est = RecursiveBayesianIntentEstimator(n_goals=4)
        est.update(np.zeros(4), np.zeros(4))
        est.reset()
        np.testing.assert_allclose(est.posterior, np.full(4, 0.25))

    def test_closer_goal_gains_mass(self):
        """Directionality: a smaller angle and shorter path are evidence FOR a goal."""
        est = RecursiveBayesianIntentEstimator(n_goals=3)
        est.update(np.array([5.0, 150.0, 160.0]), np.array([2.0, 20.0, 22.0]))
        self.assertEqual(est.map_index, 0)
        self.assertGreater(est.posterior[0], est.posterior[1])
        self.assertGreater(est.posterior[0], est.posterior[2])

    def test_transition_matrix_is_stochastic(self):
        est = RecursiveBayesianIntentEstimator(n_goals=4, delta=0.2)
        matrix = est._transition
        np.testing.assert_allclose(matrix.sum(axis=0), np.ones(4))
        self.assertAlmostEqual(matrix[0, 0], 0.8)
        self.assertAlmostEqual(matrix[0, 1], 0.2 / 3.0)

    def test_convergence_reaches_the_true_goal(self):
        traj = make_clean_convergence()
        est = RecursiveBayesianIntentEstimator(n_goals=traj.n_goals)
        posteriors = est.run(traj.angles, traj.paths)
        self.assertEqual(int(posteriors[-1].argmax()), traj.true_goal_by_epoch[0])
        self.assertGreater(posteriors[-1].max(), 0.9)

    def test_no_rounding_in_the_recursion(self):
        """The original implementation rounded to 2 d.p. on every tick, quantising
        the very quantity this substrate observes. Small masses must survive."""
        est = RecursiveBayesianIntentEstimator(n_goals=4)
        for _ in range(30):
            est.update(np.array([2.0, 170.0, 175.0, 178.0]),
                       np.array([1.0, 24.0, 24.5, 25.0]))
        losers = np.sort(est.posterior)[:3]
        self.assertTrue(np.all(losers > 0.0), "posterior mass was quantised to zero")
        self.assertLess(losers.max(), 0.01, "expected sub-1% masses to be representable")

    def test_kl_and_binary_entropy(self):
        p = np.array([0.5, 0.5])
        self.assertAlmostEqual(kl_divergence(p, p), 0.0, places=12)
        self.assertAlmostEqual(float(binary_entropy(0.5)), 1.0, places=10)
        self.assertAlmostEqual(float(binary_entropy(1.0)), 0.0, places=6)


class TestMonotoneIntegrationBoundary(unittest.TestCase):
    """The structural finding that chose the scorer.

    AST integrates with a running maximum. A posterior can un-resolve. Whether the
    two are compatible depends entirely on the SHAPE of the scorer:

      decisiveness  s = 1 - H_b(p)   NON-monotone in p; collapses at p = 0.5,
                                     which is where a contested hypothesis lives.
                                     Churn cannot ratchet it. SI keeps firing.
      veridical     s = p (or 1-p)   MONOTONE in p. A transient swing is locked in
                                     by the running maximum. SI is suppressed.
    """

    @classmethod
    def setUpClass(cls):
        traj = make_ambiguous_recurrence()
        est = RecursiveBayesianIntentEstimator(n_goals=traj.n_goals)
        cls.posteriors = est.run(traj.angles, traj.paths)
        cls.true = traj.true_goal_by_epoch[0]
        cls.contested = [0, 1]

    def _integrated(self, scores):
        return np.maximum.accumulate(scores, axis=0)[-1]

    def test_decisiveness_does_not_ratchet_under_churn(self):
        scores = 1.0 - binary_entropy(self.posteriors)
        final = self._integrated(scores)[self.contested]
        saturation = float((1.0 - final).mean())
        self.assertGreater(
            saturation, 0.5,
            "decisiveness ratcheted: contested hypotheses look resolved, SI will be "
            "suppressed in the scenario built to produce a stall",
        )

    def test_veridical_does_ratchet_under_churn(self):
        scores = np.where(
            np.arange(self.posteriors.shape[1])[None, :] == self.true,
            self.posteriors, 1.0 - self.posteriors,
        )
        final = self._integrated(scores)[self.contested]
        saturation = float((1.0 - final).mean())
        self.assertLess(
            saturation, 0.35,
            "if this passes, veridical no longer ratchets and the scorer rationale "
            "in binding_boir.py needs revisiting",
        )

    def test_decisiveness_detects_the_stall_and_veridical_underreports_it(self):
        traj = make_ambiguous_recurrence()
        means = {}
        for scorer in ("decisiveness", "veridical"):
            records = run_boir_ast(traj, scorer=scorer)[0]
            means[scorer] = float(np.mean([r["si"] for r in records]))
        self.assertGreater(means["decisiveness"], 0.30)
        self.assertGreater(
            means["decisiveness"], 2.0 * means["veridical"],
            "the ratchet should suppress veridical SI by roughly 2.5x",
        )

    def test_decisiveness_wins_pooled_auc(self):
        pooled = {}
        for scorer in ("decisiveness", "veridical"):
            records = [
                r
                for traj in all_scenarios().values()
                for recs in run_boir_ast(traj, scorer=scorer).values()
                for r in recs
            ]
            pooled[scorer] = separation_metrics(records)["auc"]
        self.assertAlmostEqual(pooled["decisiveness"], 1.0, places=6)
        self.assertGreater(pooled["decisiveness"], pooled["veridical"])


class TestRegimes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = {
            name: run_boir_ast(traj) for name, traj in all_scenarios().items()
        }

    def _mean_si(self, name, epoch=0):
        return float(np.mean([r["si"] for r in self.runs[name][epoch]]))

    def test_ambiguous_recurrence_is_the_loudest(self):
        stall = self._mean_si("ambiguous_recurrence")
        for control in ("clean_convergence", "intent_switch", "confidently_wrong"):
            self.assertGreater(stall, 3.0 * self._mean_si(control))

    def test_clean_convergence_resolves_quietly(self):
        records = self.runs["clean_convergence"][0]
        self.assertGreater(records[-1]["knowledge_mean"], 0.85)
        self.assertTrue(all(r["si"] < 0.20 for r in records))

    def test_intent_switch_re_arms_pe_after_the_reset(self):
        first, second = self.runs["intent_switch"][0], self.runs["intent_switch"][1]
        # AST resets, so the second epoch starts from an unresolved schema again.
        self.assertGreater(second[0]["pe_total"], first[-1]["pe_total"])
        # BOIR does not reset, yet both epochs still resolve.
        self.assertGreater(first[-1]["knowledge_mean"], 0.85)
        self.assertGreater(second[-1]["knowledge_mean"], 0.85)

    def test_both_epochs_identify_their_own_true_goal(self):
        for epoch in (0, 1):
            records = self.runs["intent_switch"][epoch]
            correct = [r["ground_truth"]["map_correct"] for r in records]
            self.assertGreater(sum(correct) / len(correct), 0.75)


class TestConfidentlyWrongControl(unittest.TestCase):
    """The limitation this substrate exists to expose.

    The estimator adjudicates the hypothesis space cleanly and efficiently, and is
    wrong in every window. SI is low, and SI is RIGHT to be low: there is no
    friction. AST can see whether an inference process is still learning. It cannot
    see whether what it learned is true.
    """

    @classmethod
    def setUpClass(cls):
        cls.records = run_boir_ast(make_confidently_wrong())[0]

    def test_the_estimator_is_wrong_in_every_window(self):
        correct = [r["ground_truth"]["map_correct"] for r in self.records]
        self.assertEqual(sum(correct), 0)

    def test_the_deployable_monitor_reports_no_friction(self):
        mean_si = float(np.mean([r["si"] for r in self.records]))
        self.assertLess(mean_si, 0.20)
        self.assertGreater(self.records[-1]["knowledge_mean"], 0.85)

    def test_it_is_not_labelled_a_stall(self):
        # Deliberate. It is not a stall. It is a different failure, and the friction
        # channel is not the instrument that detects it.
        self.assertTrue(all(not r["ground_truth"]["stall"] for r in self.records))

    def test_the_veridical_annotation_does_see_it(self):
        # Truth is available only as an annotation, never as a scorer input.
        self.assertLess(self.records[-1]["ground_truth"]["p_true"], 0.05)


class TestBoirBindingContract(unittest.TestCase):
    def test_embedding_is_wider_than_any_per_dimension_log(self):
        """SI's orthogonal novelty is computed against the span of prior evidence
        for a dimension. If the embedding were narrower than the number of windows
        one dimension can accumulate, its subspace would reach full rank, eta would
        collapse to zero, and the geometry would drop out of the signal."""
        traj = make_ambiguous_recurrence()
        est = RecursiveBayesianIntentEstimator(n_goals=traj.n_goals)
        windows = build_windows(traj, est)[0]
        width = embed_inference_window(windows[0]).size
        self.assertEqual(width, 4 * len(GOAL_NAMES) + 6)
        self.assertGreater(width, len(windows))

    def test_records_carry_eta_for_phase_space(self):
        records = run_boir_ast(make_ambiguous_recurrence())[0]
        self.assertTrue(any(r["eta"] for r in records))
        for record in records:
            for eta in record["eta"].values():
                self.assertGreaterEqual(eta, 0.0)
                self.assertLessEqual(eta, 1.0)

    def test_schema_is_the_hypothesis_set(self):
        binding = make_boir_binding()
        self.assertEqual(binding.schema, GOAL_NAMES)

    def test_unknown_scorer_rejected(self):
        with self.assertRaises(ValueError):
            make_boir_binding(scorer="wishful")

    def test_epochs_produce_separate_sessions(self):
        runs = run_boir_ast(make_intent_switch())
        self.assertEqual(sorted(runs), [0, 1])
        for epoch, records in runs.items():
            self.assertEqual(records[0]["t"], 1)   # each epoch is its own session
            self.assertEqual(records[0]["si"], 0.0)


if __name__ == "__main__":
    unittest.main()
