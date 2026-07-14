# Validation status

## What is validated now

`python -m unittest discover -s tests -v` — 30 tests, no network, no Ollama, no
MuJoCo. They cover:

- monotone state integration and gain/upsilon consistency;
- finite, bounded PE and SI; eta and rho complementary and in [0, 1];
- binding-contract enforcement (schema, score range, target, embedding norm);
- ordinal regime separation on both substrates;
- suppression of the saturated-hold control, which is the key negative control;
- identity of the linguistic and kinematic trajectory record shapes;
- numerical equivalence of the refactored SI against the pre-refactor
  implementation vendored in `tests/reference_si.py`.

The regime tests are **ordinal**. They assert that stalls rank above controls.
They do not pin absolute SI values, because those depend on the embedding map,
and the linguistic arm in CI runs on hash vectors.

Observed under hash embeddings (smoke path, not the experiment):

```text
hiker      efficient 0.004 | agent_repetition 0.124 | interviewee_degradation 0.231
kinematic  clean     0.051 | orbit_stall      0.399 | saturated_hold          0.040
```

## What remains before publication

1. On a machine running Ollama:

   ```bash
   python precompute_embeddings.py
   python check_embedding_separation.py
   ```

   The gate script reports the anisotropy of the archive and whether the three
   linguistic regimes still separate under real representations. Do not report
   hiker numbers until it passes.

2. Commit `data/hiker_embeddings_qwen3.npz` and its manifest, so that a cloner
   without Ollama can reproduce the headline experiment rather than only the
   smoke path.

3. Rerun `python exp1_evaluation.py` and replace any hash-embedding figures.

The kinematic substrate is unaffected by all of the above: its embedding map is
the declared numeric state representation, so it is fully reproducible today.
