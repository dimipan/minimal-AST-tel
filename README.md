# Acquisition-State Telemetry — public synthetic demonstration

Acquisition-State Telemetry (AST) is a substrate-agnostic monitor for an
evolving information-acquisition process. It does not choose actions. It turns a
declared acquisition state into two live channels:

- **Bayesian Progress Estimator (`PE^B`)**: residual task-relevant acquisition pressure.
- **Geometric Stalling Index (`SI^perp`)**: recurrent evidence with weak gain, suppressed as the relevant schema becomes resolved.

The **episodes and task data** in this repository are synthetic. The linguistic
experiment uses real embeddings from the local Ollama model
`qwen3-embedding`; synthetic/hash embeddings are available only as an explicit
smoke-test fallback.

This repository does not contain the paper's participant data, complete
experimental suite, model checkpoints, private paths, or under-review results.

## Architecture

Each substrate declares a binding:

```text
B = (M, Y, S, e)
```

- `M`: finite task schema;
- `Y`: evidence space;
- `S(y_t)`: bounded per-schema completeness scores;
- `e(y_t)`: unit-normalised evidence representation.

Everything after that boundary is shared. `substrate_binding.py` performs
monotone state integration and invokes the common equations in
`telemetry_tools_geometric.py` with the shared values in
`calibration_manifest.json`.

“Shared” means the monitor settings are held constant across the included
substrates. It does not mean the source files are cryptographically locked.

## Included substrates

Three bindings: a language stream, a physical task, and **another inference
process**. The third is what turns "substrate-agnostic" from a slogan into a
claim with a test.

### Synthetic SAR hiker dialogue

`corpus_hiker.py` contains a fully fictional search-and-rescue interview with
an eight-category schema and three deterministic trajectories:

- `efficient`;
- `agent_repetition`;
- `interviewee_degradation`.

The default representation is a precomputed `qwen3-embedding` vector for each
question-answer evidence item. The archive is stored as NPZ and loaded with
`allow_pickle=False`.

### Synthetic kinematic pick-and-place

`binding_kinematic.py` is a dependency-free continuous-control analogue with
this task schema:

```text
reach -> grasp -> lift -> place
```

It provides clean execution, orbit-stall, and saturated-hold trajectories and
emits the same `ast-tau-v1` records as the dialogue substrate.

### BOIR intent estimator — AST over another inference process

`binding_boir.py` is the substrate that makes the substrate-agnosticism claim
non-trivial. The monitored plant is not a task. It is a **recursive Bayesian
estimator** inferring which of four goals an operator intends, from angle and
path-length observations. AST asks a question about the *estimator*:

> Is incoming evidence still resolving intent, or is the estimator processing
> recurrent evidence without epistemic progress?

```text
M : goal hypotheses {h_1..h_4}; dimension i = "hypothesis i adjudicated",
    by confirmation OR elimination -- ruling a goal out is acquisition
Y : a window of K=5 estimator ticks (posterior dynamics + the observations
    driving them)
S : decisiveness, s_i = 1 - H_b(p_i)  -- deployable, needs no ground truth
e : 22-d inference-dynamics signature (posterior mean/delta, angle and path
    evidence, entropy level and trend, KL innovation, MAP-switch count)
```

Everything is generated in closed form from a seed. No robot, no simulator, no
logs, no data directory.

```bash
python exp_boir_evaluation.py
```

| scenario | mean SI | completeness | MAP correct |
|---|---|---|---|
| clean convergence | 0.078 | 0.948 | 10/12 |
| **ambiguous recurrence** | **0.463** | 0.621 | 5/12 |
| intent switch (epoch 1 / 2) | 0.056 / 0.060 | 0.932 / 0.937 | 8/8, 7/8 |
| confidently wrong | 0.077 | 0.935 | **0/12** |

Pooled AUC **1.000**, perfect rank separation.

**Epochs.** Within an epoch the latent goal is fixed, so evidence accumulation
about it is legitimately monotone. Across an intent change it is not. So BOIR runs
*continuously* across the boundary — its carry-over and lag are the phenomenon —
while AST *resets*, as one session per epoch. This needs no change to the shared
runner.

**Multi-dimension gain.** A single posterior update can confirm one hypothesis
while eliminating another, so several schema dimensions gain in one exchange. The
aggregate-gain dampening declared in the manifest is not an optimisation here; it
is a correctness requirement, and it was already in place.

## Two findings from the BOIR substrate

Both are boundary conditions, and both are stated here because a reviewer would
otherwise find them.

### 1. The scorer must be non-monotone in the monitored state

AST integrates with a running maximum, `upsilon_i(t) = max(upsilon_i(t-1),
s_i(t))`. A posterior can **un-resolve**. Whether the two are compatible depends
entirely on the *shape* of the scorer. Measured on `ambiguous_recurrence`, on the
two contested hypotheses:

| scorer | monotone in p? | final `upsilon` | saturation weight `(1 - upsilon)` | pooled AUC |
|---|---|---|---|---|
| decisiveness `1 - H_b(p)` | **no** | 0.367 / 0.310 | **0.661** | **1.000** |
| veridical `p` / `1 - p` | yes | 0.745 / 0.774 | 0.241 | 0.854 |

Decisiveness collapses to zero at `p = 0.5` — exactly where a contested hypothesis
lives — so churn cannot ratchet the running maximum, the saturation weight stays
high, and SI keeps firing. Veridical resolvedness is monotone in `p`, so a
transient swing is **locked in** by the maximum: `upsilon` records a peak the
estimator never sustained, and SI is suppressed ~2.6x in the very scenario built to
produce a stall.

So the deployable scorer is the default, and truth enters as a ground-truth
**annotation** (`p_true`, `map_correct`) rather than as a scorer input.
`--scorer veridical` remains available as a diagnostic and prints a warning.

Generalised: **monotone integration is appropriate where acquisition is
irreversible. A process that can lose ground needs a scorer bounded away from 1
while its state is contested.** That is a real constraint on where AST applies, and
it is not visible in a substrate where progress cannot be undone.

### 2. A truth-free friction channel cannot certify correctness

In `confidently_wrong` the estimator adjudicates the hypothesis space to
completeness **0.935** at mean SI **0.077** — no friction, no stall — and its MAP
hypothesis is wrong in **12 of 12** windows.

SI is *correct* to be low. The estimator is not stuck. It is confident and
mistaken, and those are different failures.

> **AST can tell you whether an inference process is still learning. It cannot tell
> you whether what it learned is true.**

The veridical annotation sees it (`p(true) = 0.007`). The deployable channel cannot,
and no amount of threshold tuning will change that.

## Set up the Qwen3 hiker experiment

Install and start Ollama, then make the embedding model available locally:

```bash
ollama pull qwen3-embedding
ollama serve
```

In another terminal:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[embeddings]'
python precompute_embeddings.py
python check_embedding_separation.py    # gate; see below
python exp1_evaluation.py
```

`check_embedding_separation.py` exists because transformer embedding spaces are
anisotropic: cosine similarities bunch high rather than near zero, which lifts
`SI^perp` on every exchange, including the efficient control. Hash vectors are
near-orthogonal by construction and cannot reveal this. The script reports the
cosine spread of the archive and whether the three regimes still separate. If
they separate in mean but not against the fixed threshold, report mean-SI
separation and drop the threshold table — do not retune the threshold.

`precompute_embeddings.py` calls the local endpoint used by the research code:

```text
http://localhost:11434/api/embeddings
```

It writes:

```text
data/hiker_embeddings_qwen3.npz
data/hiker_embeddings_qwen3.manifest.json
```

A different endpoint, model, or output location can be supplied explicitly:

```bash
python precompute_embeddings.py \
  --model qwen3-embedding \
  --ollama-url http://localhost:11434/api/embeddings \
  --output data/hiker_embeddings_qwen3.npz
```

The archive itself is not included in this draft because it must be generated
by a machine running Ollama. Once generated, it can be committed to the public
repository if redistribution of those model outputs is acceptable for the
project.

## Smoke tests without Ollama

Hash vectors exist only to test the plumbing and CI:

```bash
python exp1_evaluation.py --hash-embeddings
python -m unittest discover -s tests -v
```

Results from this mode must not be presented as the hiker experiment. The JSONL
metadata records `embedding_backend: hash-smoke` so the two modes cannot be
confused accidentally.

## Kinematic example

```bash
python exp_kinematic_evaluation.py
```

## Shared trajectory contract

Both substrates emit `ast-tau-v1.1` JSONL. Each exchange record carries:

```text
t                target exchange index
i_t              target schema category
upsilon          per-category completeness
delta_upsilon    per-category gain this exchange
PE / PE_total    Bayesian progress estimator
SI               geometric stalling index
eta              per-category orthogonal novelty  (phase-space: NOVELTY axis)
rho              per-category geometric membership, = 1 - eta
dampening        D(t) = max(eta_min, 1 - lambda * sum_i delta_upsilon_i)
gt               constructed ground-truth annotations
```

`eta` against `delta_upsilon` is the novelty-versus-yield plane. The phase-space
package should consume this contract only and should not import either substrate.

### Dampening is aggregate, and declared

`D(t)` uses the **aggregate** gain summed over all schema dimensions, not the
target dimension's gain alone. The two coincide only when exactly one dimension
gains per exchange; both substrates here violate that (a free-recall answer
resolves several categories at once; the grasp frame resolves `grasp` and `lift`
together). The mode is declared in `calibration_manifest.json` as
`dampening_gain: aggregate_delta_upsilon`, asserted at runner construction, and
pinned by `tests/test_frozen_core.py`.

## Tests

```bash
python -m unittest discover -s tests -v
```

Thirty tests, no network and no optional dependencies. They cover state
monotonicity, telemetry bounds, binding-contract enforcement, ordinal regime
separation on both substrates, and the identity of the two trajectory schemas.
`tests/test_frozen_core.py` additionally pins `SI^perp` against
`tests/reference_si.py`, the pre-refactor implementation vendored verbatim, so
that exporting `eta`/`rho` provably did not move any number.

## Finding: SI's ordering is universal, its scale is not

This repository ships a result, not just a demo.

The same three SAR trajectories were scored under two evidence representations —
96-d hash vectors and 4096-d `qwen3-embedding`:

| representation | median pairwise cosine | mean SI (stall) | mean SI (clean) | AUC |
|---|---|---|---|---|
| hash, 96-d | 0.012 | 0.217 | 0.008 | **1.000** |
| qwen3, 4096-d | 0.459 | 0.174 | 0.008 | **1.000** |

Rank separation is perfect under both: every stall exchange scores above every
clean exchange. But the absolute scale moves, and it moves for a structural
reason. SI is normalised:

```text
SI(t) = sum_i rho_i(t)^2 * D(t) * (1 - upsilon_i(t))  /  sum_i rho_i(t)
```

Hash vectors are near-orthogonal by construction, so a stall's evidence has
membership `rho ~ 0` on every non-target dimension and the denominator stays
small. A real embedding space is not orthogonal: `rho` is non-trivial on other
dimensions too, the denominator grows, and SI compresses — without the numerator
or the ordering changing.

The consequence is a calibration rule, not a caveat:

> **`SI^perp`'s ordering is a property of the monitor. Its absolute scale is a
> property of the evidence representation. A stall threshold fitted under one
> embedding map does not transfer to another.**

So `theta` is **not** in the shared calibration block. It lives in
`per_binding_calibration`, and:

- the **kinematic** binding declares `theta = 0.20` — its representation is the
  declared numeric state vector, its SI scale is stable, and the pooled sweep
  shows a wide plateau (`theta` in 0.20–0.40 gives precision 1.00, FPR 0.00);
- the **SAR** binding declares **no threshold at all**. AUC is 1.000 on both
  stall regimes, but 9-exchange sessions resolve false-positive rate only to 1/9,
  which is too coarse to fit `theta` honestly. It reports AUC and mean-SI
  separation, and prints the `theta = 0.20` operating point anyway — including
  the places it fails — so the failure is visible rather than hidden.

Both evaluations print a pooled threshold sweep so a reader can see where an
operating point would sit without the author having chosen one. **The threshold
was not retuned to rescue the linguistic result.**

## Reading the figures

Every evaluation writes four kinds of figure. They follow rules that exist
because the obvious version of each is misleading:

- **Per-condition time series** — conditions from the same substrate share axis
  limits and a shaded `theta` band. Per-condition autoscaling makes an efficient
  control's SI noise look like a stall.
- **Global AST telemetry trajectory** (`*_telemetry_trajectory.png`) —
  completeness x PE x SI, with a translucent `SI = theta` plane. Time is encoded
  by colour, arrows and step labels, and marker area grows with consecutive
  no-gain occupancy, so dwell is visible. Deliberately *not* called a phase
  portrait: the axes are telemetry channels, not a state and its derivative.
- **Regime portrait** (`regime_portrait.png`) — mean gain x SI x PE. This is the
  view that discriminates. A saturated hold and an unresolved stall both sit at
  zero gain and are separated *only* by SI and PE together, which is the case for
  reporting both channels rather than either alone.
- **Evidence geometry** (`evidence_geometry.png`) — one PCA basis fitted on the
  union of all conditions, with all panels sharing that basis *and* one viewport;
  separately fitted projections are not comparable. Recurrence links are computed
  by cosine in the **full** embedding space and only drawn in the projection, and
  candidates within 3 exchanges are excluded, because adjacency is not
  recurrence: a smooth trajectory has cosine > 0.99 with the frame it just left.
  In the kinematic substrate the clean and saturated runs traverse the space
  while the orbit stall never leaves one region.

Link *counts* in the evidence-geometry view are themselves
representation-dependent, for the same reason SI's scale is — read the spatial
extent, not the tally.

## Repository map

```text
telemetry_tools_geometric.py    shared PE/SI equations
calibration_manifest.json       shared monitor settings
substrate_binding.py            B=(M,Y,S,e), runner, JSONL contract
corpus_hiker.py                 synthetic linguistic binding
binding_kinematic.py            synthetic continuous-control binding
bayesian_intent.py              recursive Bayesian intent estimator (BOIR)
synthetic_intent_trajectories.py deterministic angle/path streams, 4 scenarios
binding_boir.py                 AST-over-estimator binding
exp1_evaluation.py              qwen3 hiker evaluation (threshold-free headline)
exp_kinematic_evaluation.py     kinematic evaluation
exp_boir_evaluation.py          BOIR estimator evaluation
precompute_embeddings.py        Ollama qwen3-embedding precomputation
check_embedding_separation.py   gate: does qwen3 preserve regime separation?
evaluation_utils.py             detector metrics
plotting.py                     static plots
visualisation.py                substrate-independent telemetry-trajectory renderer
examples/run_both_substrates.py one trajectory from each substrate
optional/ppo_ast_demo.py        optional control integration
tests/                          54 tests; no network, no Ollama, no MuJoCo
```

## Scope

- Synthetic refers to the task data and constructed regimes, not to the default linguistic embedding model.
- Ground-truth stalls are constructed by design.
- `SI^perp` is a low-yield recurrence measurement under the declared binding, not a causal diagnosis.
- `PE^B` and `SI^perp` are telemetry. Alert, reward, truncation, or intervention policies are downstream choices.
- The optional PPO example is an integration pattern, not a performance claim.

## License

Code and synthetic task data are released under the MIT License.
