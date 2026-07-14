# Public release checklist

## Done

- [x] All shipped scenarios are synthetic and labelled as such.
- [x] No participant data, private corpora, checkpoints, or scorer outputs.
- [x] No absolute machine paths.
- [x] Embedding archives use NPZ with `allow_pickle=False`; no pickles.
- [x] Hash embeddings restricted to smoke tests and CI, and tagged in the JSONL
      metadata as `embedding_backend: hash-smoke`.
- [x] Linguistic and non-linguistic examples emit the same `ast-tau-v1.1` schema.
- [x] `eta` / `rho` / `dampening` exported, so phase space reads novelty against
      yield without re-instrumenting the frozen core.
- [x] Aggregate-gain dampening declared in the manifest, asserted at runner
      construction, and pinned by a test.
- [x] Refactored SI pinned against the pre-refactor implementation.
- [x] Shared monitor calibration in one manifest.
- [x] 54 tests; cold clone runs with `pip install -e .` and no optional deps.
- [x] Third substrate (BOIR intent estimator) added: AST over another
      inference process, fully synthetic, no data directory at all.
- [x] BOIR scorer choice justified empirically (monotone-integration
      boundary) and pinned by tests, not asserted.
- [x] The confidently-wrong limitation is stated in the README and pinned
      by a test, rather than left for a reviewer to find.
- [x] CI workflow: tests plus all three examples on 3.10 and 3.12.
- [x] Optional PPO and embedding-regeneration dependencies isolated from the
      core install.

## Before making the repository public

- [ ] Generate `data/hiker_embeddings_qwen3.npz` on a machine running Ollama.
- [ ] Run `python check_embedding_separation.py` and act on the verdict.
- [ ] Commit the embedding archive so a cloner can reproduce the hiker
      experiment without Ollama.
- [ ] Rerun the hiker figures and summaries with qwen3 embeddings; delete any
      hash-embedding outputs.
- [ ] Fill in the author name in `LICENSE` and add citation metadata.
- [ ] Confirm redistribution of qwen3-embedding model outputs is acceptable.
- [ ] Add phase-space code only after its interface is fixed against
      `ast-tau-v1.1`.
