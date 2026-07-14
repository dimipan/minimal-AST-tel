# Acquisition-State Telemetry

**A monitor that watches a process acquire information, and tells you when it stops making progress.**

It doesn't choose actions. It reads two channels:

- **PE** — how much of the task is still unresolved.
- **SI** — whether evidence is repeating itself without producing anything new.

The point of this repository is that the *same equations* watch three completely different things: a witness interview, a robot arm, and a Bayesian estimator. Only a thin adapter changes.

All data here is synthetic. Everything runs in under a minute with no GPU, no simulator, and no network.

```bash
pip install -e .
python exp_kinematic_evaluation.py     # robot arm
python exp_boir_evaluation.py          # intent estimator
python -m unittest discover -s tests   # 54 tests
```

---

## How it works

Each substrate declares a **binding** — four things, and nothing else:

```text
B = (M, Y, S, e)

M  what counts as "done"        a list of categories to resolve
Y  what evidence looks like     an utterance, a robot frame, a posterior window
S  how to score it              evidence -> how resolved is each category, 0..1
e  how to represent it          evidence -> a vector
```

Everything after that line is shared and unchanged: the state integrator, the PE and SI equations, the calibration constants. Write those four things and the monitor works on your process. That's the whole claim, and the three substrates below are the test of it.

Both channels come from one place — `telemetry_tools_geometric.py` — and every substrate writes the same JSONL trajectory format.

---

## The three substrates

### 1. Search-and-rescue interview (language)

An interviewer questions a witness about a missing hiker. Eight things need establishing: location, time, description, equipment, and so on. Three runs:

| run | what happens | mean SI |
|---|---|---|
| efficient | every question lands new information | **0.006** |
| agent repetition | the interviewer keeps re-asking a question already answered | 0.099 |
| interviewee degradation | the witness has nothing left to give and says so, four different ways | 0.085 |

Evidence is embedded with a real `qwen3-embedding` model. The text is fictional; the embeddings are not.

**What this means.** The monitor separates the two stall types *perfectly* — every stalled turn scores above every productive turn, in both runs (AUC 1.000). But look at how small the numbers are. That's the finding, and it has a section of its own below.

Also notice that "interviewee degradation" scores higher than "agent repetition" even though the repetition is word-for-word identical. That's correct. Re-asking about *location* — already 78% resolved — is barely a stall, because there was little left to gain. Hammering *equipment* — stuck at 20% — is a real one. The monitor weights a stall by what's still at stake.

---

### 2. Robot pick-and-place (physical task)

A four-step task: reach → grasp → lift → place. Pure NumPy kinematics, no MuJoCo. Three runs:

| run | what happens | mean SI | at θ=0.20 |
|---|---|---|---|
| clean | the arm completes the task | 0.051 | — |
| **orbit stall** | the arm circles the object without ever grasping it | **0.399** | recall 0.94, precision 1.00 |
| **saturated hold** | the arm finishes, then hovers at the goal doing nothing | 0.040 | no alarm |

**What this means.** Compare the last two rows. **Both are "the robot is doing nothing new."** Both would trip a naive repetition detector. The monitor fires on one and stays silent on the other — because in the orbit the task is *unfinished*, and in the hold it's *done*.

That's the whole reason SI is multiplied by `(1 − completeness)`. An idle robot that has finished its job is not stalled. `saturated_hold` is the single most important control in this repository: it's the one that proves SI is measuring **stalling**, not merely **repetition**.

Here the fixed threshold works: θ anywhere from 0.20 to 0.40 gives precision 1.00 and zero false positives.

---

### 3. Bayesian intent estimator (another inference process)

This one isn't a task at all. A recursive Bayesian estimator watches an operator move and infers which of four goals they're heading for. AST watches **the estimator**: is evidence still resolving intent, or is the estimator spinning?

| run | what happens | mean SI | estimator right? |
|---|---|---|---|
| clean convergence | evidence steadily favours one goal | 0.078 | 10/12 ✓ |
| **ambiguous recurrence** | two goals stay equally plausible forever | **0.463** | 5/12 |
| intent switch | the operator changes their mind halfway | 0.056 / 0.060 | 8/8, 7/8 ✓ |
| **confidently wrong** | evidence cleanly favours the **wrong** goal | 0.077 | **0/12** ✗ |

Pooled AUC **1.000**, with real margin: the quietest stall (0.191) still beats the loudest clean exchange (0.140).

**What this means — and read the last row twice.**

In `confidently_wrong`, the estimator resolves the question cleanly and confidently. Completeness reaches **0.935**. SI stays at **0.077** — no friction, no alarm. And the estimator is **wrong in every single window**.

SI is *right* to be quiet. The estimator isn't stuck. It's confident and mistaken, and those are different failures.

> **AST tells you whether a process is still learning. It cannot tell you whether what it learned is true.**

That's a hard limit, it's demonstrated here rather than hedged around, and no threshold tuning will move it.

---

## Two findings you should know before trusting a number

### SI's *ranking* is reliable. Its *scale* is not.

We scored the same three interview runs under two different embedding models:

| embeddings | mean SI (stalls) | mean SI (clean) | AUC |
|---|---|---|---|
| hash vectors, 96-d | 0.217 | 0.008 | **1.000** |
| qwen3, 4096-d | 0.174 | 0.008 | **1.000** |

Stalls beat clean turns every time under both. But the absolute numbers move, because SI is normalised by how much the evidence "belongs" to each category — and different embedding spaces spread evidence differently.

**So a stall threshold tuned on one representation does not transfer to another.** θ lives in `per_binding_calibration`, not in the shared block:

- **robot** and **estimator** declare θ = 0.20 — their representations are fixed numeric vectors, and the sweep shows a wide safe plateau.
- **the interview declares no threshold at all.** Its ranking is perfect, but 9-turn sessions can only measure a false-positive rate to the nearest 1/9. That's too coarse to set a threshold honestly, so we don't set one. We report AUC and mean separation, print the θ=0.20 numbers *including where they fail*, and **we did not retune θ to make them look better.**

Every evaluation prints a threshold sweep so you can see where an operating point would sit without us having picked one for you.

### The scorer has to be able to go back down

AST records the *best* score each category has ever reached. That's fine when progress can't be undone — a fact learned, an object grasped. A Bayesian posterior can un-learn, and that breaks the assumption.

Two ways to score the estimator, on the churning `ambiguous_recurrence` run:

| scorer | recorded state | room left to stall | pooled AUC |
|---|---|---|---|
| **decisiveness** — how *settled* is this hypothesis | 0.37 | **0.66** | **1.000** |
| veridical — how *close to the truth* is it | 0.75 | 0.24 | 0.854 |

Decisiveness bottoms out when a hypothesis is a coin-flip, which is exactly where a contested hypothesis sits — so churn can't inflate it. Veridical resolvedness rises and falls with the posterior, so one lucky swing gets locked in as a permanent high-water mark, and the monitor concludes the question is settled when it isn't. SI drops 2.6× in the run built to produce a stall.

So the default scorer is the truth-free one, and truth enters only as a ground-truth *label* for evaluation. Generalised:

> **AST assumes progress is permanent. If your process can lose ground, the scorer must be one that falls when the process does.**

That's a real limit on where this applies, and you can't see it in a substrate where progress can't be undone. Which is why the estimator is in here.

---

## Reading the figures

Every evaluation writes four. Each follows a rule that exists because the obvious version misleads.

**Time series** (`<run>.png`) — all runs from one substrate share the same axes, with the θ band shaded. Autoscaling each run separately makes a clean run's noise look like a stall.

**Telemetry trajectory** (`*_telemetry_trajectory.png`) — completeness × PE × SI in 3-D, with the θ plane drawn in. Colour is time; **dots get bigger the longer a run goes without gaining anything**. A stall looks like a pile-up.

**Regime portrait** (`regime_portrait.png`) — gain × SI × PE. This is the one that discriminates:

| | gain | SI | PE |
|---|---|---|---|
| productive | > 0 | low | varies |
| **unresolved stall** | ≈ 0 | **high** | **high** |
| **saturated hold** | ≈ 0 | **low** | **low** |

A stall and a finished-and-idle process both sit at zero gain. **Only SI and PE together tell them apart** — which is why you need both channels, and why they're plotted against each other.

**Evidence geometry** (`evidence_geometry.png`) — where the evidence actually went. One PCA basis fitted across all runs, one shared viewport, so positions are comparable. Red lines mark a return to somewhere you'd already been (computed in the full space, drawn in 2-D; steps within 3 exchanges are excluded, because moving smoothly isn't the same as going back).

In the robot figure: the clean and hold runs sweep across the whole space. **The orbit stall never leaves its corner.** That's the picture of the thing.

*(Check the variance number in the caption. At 93% — the robot — you can trust fine positions. At 56% — the estimator — read the spread, not the detail. The plot says so itself.)*

---

## What's honest about this repo

- Synthetic **task data**, not synthetic embeddings. The interview uses a real embedding model.
- The core equations are unchanged from the research code, and a test replays every exchange against the original implementation to prove it.
- Constructed stalls are labelled *by scenario design*, never by a rule derived from the monitor's own output.
- A within-run AUC is suppressed when its only clean exchanges are warm-up exchanges — that number would be 1.000 for free, so we don't print it.
- Where the method fails, it says so: no threshold for the interview, no correctness guarantee for the estimator.
- The optional PPO example is an integration pattern, **not** a performance claim.

## Layout

```text
telemetry_tools_geometric.py     the PE and SI equations — shared, unchanged
substrate_binding.py             B = (M, Y, S, e), the runner, the JSONL contract
calibration_manifest.json        every constant, in one place

corpus_hiker.py                  substrate 1: interview
binding_kinematic.py             substrate 2: robot arm
bayesian_intent.py               substrate 3: the estimator being watched
synthetic_intent_trajectories.py     its four scenarios, generated from a seed
binding_boir.py                      and its binding

exp1_evaluation.py               run substrate 1   (add --hash-embeddings for no-Ollama)
exp_kinematic_evaluation.py      run substrate 2
exp_boir_evaluation.py           run substrate 3

plotting.py / visualisation.py   figures; the viewer reads any substrate's JSONL
tests/                           54 tests — no network, no Ollama, no MuJoCo
optional/ppo_ast_demo.py         wiring AST into an RL loop
```

## Running the interview experiment with real embeddings

```bash
ollama pull qwen3-embedding && ollama serve
python precompute_embeddings.py
python check_embedding_separation.py   # confirms the regimes still separate
python exp1_evaluation.py
```

Without Ollama, `--hash-embeddings` runs the same pipeline on deterministic hash vectors. It's a plumbing check, not the experiment, and the output says so on every line.

## Scope

- SI measures **low-yield recurrence under a declared binding**. It is not a diagnosis of *why*.
- PE and SI are **telemetry**. What you do with them — alert, penalise, truncate, intervene — is a separate decision.
- The full empirical study is under review. This repository is the method, on synthetic data.

MIT licensed.
