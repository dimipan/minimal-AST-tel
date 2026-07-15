# Acquisition-State Telemetry

**A monitor that watches a process acquire information, and reports when it stops making progress.**

Most ways of checking whether a system is "working" look at the output. AST looks
at the *process*: as evidence arrives, is the task actually getting resolved, or is
the system going through the motions — repeating itself, drifting, or coasting on
work it already finished? It reads two channels off the live trajectory:

- **PE** — how much of the task is still unresolved.
- **SI** — whether incoming evidence is recurring without producing anything new.

The reason this repository exists is one claim: **the same measurement works no
matter what the process is.** The equations don't know whether they're watching a
conversation, a robot, or another algorithm. To show that, the same frozen monitor
is bound to three deliberately different processes — a witness interview, a robot
arm, and a Bayesian estimator — and asked the same question of each.

Everything here is synthetic and small. That's the point: it's fully inspectable,
runs in under a minute, needs no GPU or simulator, and every result below
reproduces from a clean clone. The environments are simplified on purpose — this is
the *method*, stripped to where you can read all of it, not the full study.

```bash
pip install -e .
python exp_kinematic_evaluation.py
python exp_boir_evaluation.py
python -m unittest discover -s tests    # 78 tests
```

---

## How it works

Each process declares a **binding** — four things, nothing more:

```text
B = (M, Y, S, e)

M  what counts as "done"        a list of categories to resolve
Y  what evidence looks like     an utterance, a robot frame, a posterior window
S  how to score it              evidence -> how resolved is each category, 0..1
e  how to represent it          evidence -> a vector
```

Everything after that line is shared and identical across all three processes: the
state integrator, the PE and SI equations, the calibration constants. Write those
four things and the monitor works on your process. That is the entire claim, and
the three substrates are the test of it.

Both channels come from one file — `telemetry_tools_geometric.py` — and every
substrate writes the same JSONL trajectory format, which is what lets the
phase-space layer read all three without knowing which is which.

---

## The three substrates

### 1. Search-and-rescue interview (language)

An interviewer questions a witness about a missing hiker; eight things need
establishing. Three runs, scored with a real `qwen3-embedding` model (the text is
fictional, the embeddings are real):

| run | what happens | mean SI |
|---|---|---|
| efficient | every question lands new information | 0.006 |
| agent repetition | the interviewer re-asks a question already answered | 0.099 |
| interviewee degradation | the witness has nothing left and says so, four ways | 0.085 |

Both stall types score above the efficient baseline — perfect rank separation on
the underlying signal. But note the interview is where the *regimes differ from each
other most*, which the phase space below makes visible. Re-asking about a category
that's already 78% resolved is a weak stall; hammering one stuck at 20% is a strong
one. The monitor weights a stall by what's still at stake, so the two don't look
alike, and shouldn't.

### 2. Robot pick-and-place (physical task)

A four-step task, reach → grasp → lift → place, in pure NumPy — no MuJoCo. Three
runs:

| run | what happens | mean SI | at θ=0.20 |
|---|---|---|---|
| clean | the arm completes the task | 0.051 | — |
| **orbit stall** | the arm circles the object, never grasping | **0.399** | recall 0.94, precision 1.00 |
| **saturated hold** | the arm finishes, then hovers doing nothing | 0.040 | no alarm |

The last two rows are the whole idea in miniature. **Both are "the robot is doing
nothing new."** A naive repetition detector trips on both. The monitor fires on the
orbit — where the task is unfinished — and stays silent on the hold — where it's
done. That difference is why SI is multiplied by `(1 − completeness)`: an idle robot
that finished its job is not stalled. `saturated_hold` is the most important control
in the repo — the one that shows AST measures **stalling**, not merely **repetition**.

### 3. Bayesian intent estimator (another inference process)

Not a task at all. A recursive Bayesian estimator infers which of four goals an
operator intends; AST watches *the estimator* — is evidence still resolving intent,
or is it spinning?

| run | what happens | mean SI | estimator right? |
|---|---|---|---|
| clean convergence | evidence steadily favours one goal | 0.078 | 10/12 ✓ |
| **ambiguous recurrence** | two goals stay equally plausible forever | **0.463** | 5/12 |
| intent switch | the operator changes their mind halfway | 0.056 / 0.060 | 8/8, 7/8 ✓ |
| **confidently wrong** | evidence cleanly favours the **wrong** goal | 0.077 | **0/12** ✗ |

Pooled AUC **1.000**, with real margin (quietest stall 0.191 > loudest clean 0.140).

Read the last row twice. In `confidently_wrong` the estimator resolves cleanly and
confidently, completeness reaches **0.935**, SI stays at **0.077** — and it is wrong
in every single window. SI is *right* to be quiet: the estimator isn't stuck, it's
confident and mistaken, which is a different failure.

> **AST tells you whether a process is still learning. It cannot tell you whether
> what it learned is true.** That's a hard limit, shown here rather than hedged.

---

## Two things to know before trusting a number

### SI's *ranking* is reliable; its *scale* is not.

The same three interview runs, scored under two embedding models:

| embeddings | mean SI (stalls) | mean SI (clean) | AUC |
|---|---|---|---|
| hash vectors, 96-d | 0.217 | 0.008 | 1.000 |
| qwen3, 4096-d | 0.174 | 0.008 | 1.000 |

Stalls beat clean turns every time under both, but the absolute numbers move,
because SI is normalised by how strongly evidence belongs to each category and
different embedding spaces spread evidence differently. **So a threshold tuned on
one representation doesn't transfer to another.** θ lives in
`per_binding_calibration`, not the shared block: the robot and estimator declare
θ=0.20 (fixed numeric representations, stable scale, a wide safe plateau in the
sweep); the interview declares *no* threshold, because 9-turn sessions can only
resolve a false-positive rate to 1/9 — too coarse to set one honestly. It reports
ranking, not a threshold, and every evaluation prints a threshold sweep so you can
see where an operating point would sit without one being chosen for you.

### The scorer has to be able to go back down.

AST records the *best* score each category ever reached — fine when progress can't
be undone. A Bayesian posterior can un-learn. On the churning `ambiguous_recurrence`
run:

| scorer | recorded state | room left to stall | pooled AUC |
|---|---|---|---|
| **decisiveness** — how *settled* is this hypothesis | 0.37 | 0.66 | 1.000 |
| veridical — how *close to truth* is it | 0.75 | 0.24 | 0.854 |

Decisiveness bottoms out at a coin-flip, which is where a contested hypothesis
lives, so churn can't inflate it. Veridical rises and falls with the posterior, so a
lucky swing gets locked in as a permanent high-water mark and the monitor concludes
the question is settled when it isn't. The default scorer is the truth-free one;
truth enters only as a label for evaluation.

> **AST assumes progress is permanent. If your process can lose ground, the scorer
> must be one that falls when the process does.** A real constraint on where the
> method applies — invisible in any substrate where progress can't be undone, which
> is why the estimator is in here.

---

## The phase space

The three substrates all write the same trajectory format, so a single layer can
read any of them and place every exchange in a **regime** — no substrate, no
binding, no monitor imported. This is what the two channels are *for*: not a scalar
alarm, but a map of where a process is in its acquisition.

```bash
python phase_space_report.py outputs/boir/*.jsonl
python phase_space_report.py outputs/kinematic/*.jsonl     # same layer, no changes
python phase_space_report.py outputs/hiker/*.jsonl
```

Three telemetry quantities name the regime: **yield** (did this exchange resolve
anything), **friction** (SI), **pressure** (PE):

| regime | yield | friction | pressure | |
|---|---|---|---|---|
| productive | > 0 | — | — | resolving normally |
| **churn** | ≈ 0 | high | high | the pathological stall |
| converged | ≈ 0 | low | low | done; nothing left to do |
| quiet-incomplete | ≈ 0 | low | high | starved, not spinning |

**Why this needs both channels.** `churn` and `converged` are *both* zero-yield — a
repetition detector cannot separate them. Only friction and pressure *together*
tell "stuck with work remaining" from "finished and idle." The verdict is the
**churn fraction** — a pure count, so it transfers across substrates with no tuning.

What the runs actually produce:

| substrate | flagged pathological | not flagged |
|---|---|---|
| BOIR | ambiguous_recurrence (0.83) | clean, intent_switch, confidently_wrong |
| kinematic | orbit_stall (0.71) | clean, **saturated_hold (0.00)** |
| interview | **interviewee_degradation (0.38)** | efficient, **agent_repetition (0.00)** |

Two of these are worth pausing on, because **different substrates put their stalls
in different regimes — and that is the layer working, not failing.**

- **Kinematic `saturated_hold` → not a stall.** The robot that finished and idles is
  zero-yield forever, but pressure has drained, so it lands in `converged`, not
  `churn`. The plane gets it right on a substrate it was never tuned for.

- **Interview `agent_repetition` → not churn; `interviewee_degradation` → churn.**
  These are both "stalls" by mean SI, but the phase space splits them, correctly.
  Re-asking about location (already 78% resolved) is damped by the `(1 − completeness)`
  term and reads as `converged` — the schema was nearly closed, so there's little
  friction to register. Degradation hammers equipment (stuck at 20%) and reads as
  `churn`. The phase space is *stricter* than mean SI, and the extra structure is
  the interview substrate telling you these two stalls are not the same animal.

---

## The latent-absorption view

`phase_space.py` collapses recurrence into one friction axis. `latent_absorption.py`
opens it back up. SI is built from two primitives — **orthogonal novelty** (is this
exchange exploring new latent ground?) and **yield** — and both are already exported
per exchange. Plot them against each other and you get four quadrants, a finer
question than "is this a stall?": *what kind* of non-progress is this?

```bash
python latent_absorption_report.py outputs/boir/*.jsonl
python latent_absorption_report.py outputs/kinematic/*.jsonl
python latent_absorption_report.py outputs/hiker/*.jsonl
```

|  | high yield | low yield |
|---|---|---|
| **high novelty** | productive expansion (new ground, it pays) | **unproductive expansion** (new ground, wasted — drift) |
| **low novelty** | useful recurrence (revisiting, still pays) | **latent absorption** (revisiting, exhausted — stuck) |

The plane needs both axes, and there's a control that proves it: take two runs with
the *same* low-novelty geometry but *different* yield, and yield alone flips them
between useful recurrence and latent absorption. That paired-decoy check **passes on
real trajectories** in both BOIR and kinematic.

And here the three substrates genuinely diverge — which is the most informative
result in the repository:

- **BOIR / kinematic stalls → latent absorption.** The estimator spinning on two
  goals, and the arm orbiting the object, both revisit exhausted ground: low
  novelty, low yield. Absorption fraction 0.83 and 0.75.

- **Interview `interviewee_degradation` → unproductive expansion, *not* absorption.**
  This is the subtle one. The witness's dead-end answers ("nothing else comes back
  to me", "only the same grey pack") are *semantically fresh text* — high novelty —
  that resolves nothing. High novelty + low yield is **unproductive expansion**: the
  process is verbally exploring while epistemically stuck. The phase space called
  this run a stall (churn); the absorption view says the *kind* of stall is
  novelty-masked, not a simple loop. Both are right, and together they say something
  neither says alone.

This is the payoff of separating the two views. On a physical task, being stuck
looks like sitting still (absorption). In language, being stuck can look like
fluent motion that goes nowhere (unproductive expansion). Same monitor, same
primitives — the substrate decides which face a stall wears.

**One honest boundary.** The absorption quadrant is purely geometric — low novelty,
low yield. A robot that *finishes* and idles also shows both, so absorption alone
flags kinematic `saturated_hold` (0.26) even though it isn't stuck. Only residual
pressure (PE) separates "done" from "stuck," which is exactly what `phase_space.py`
adds, correctly calling that hold `converged`. Read the two views together: latent
absorption names the *kind* of non-progress; phase space says whether the schema is
still *open*.

*(The quadrant partition is also the natural place to state later guarantees:
membership is a set defined on two telemetry primitives, so properties like "a
trajectory that enters latent absorption and stays N steps satisfies X," or "an
intervention that raises novelty must cross the absorption/expansion boundary," are
statable against it — which they are not against a scalar. That is where this line
of work goes next.)*

---

## Why this repository exists

The bet is that **process-level observability is a substrate-general idea**: whether
a system is a dialogue agent, a controller, or another model, "is the computation
still resolving the task, or has it entered low-yield recurrence, committed
prematurely, or begun coasting" is answerable from the acquisition trajectory alone,
with the measurement kept strictly separate from any downstream policy. The three
toy substrates exist to make that claim concrete and falsifiable on something you
can read end to end in an afternoon.

They are simplified, and simplified substrates show simplified behaviour — that a
stall wears a different face on each one is a feature of the phenomenon, not a fault
in the pipeline. The same frozen monitor, unchanged, produces coherent and
*distinct* readings on language, control, and inference, and the phase space
organises those readings into regimes that transfer across all three.

**Future work** is refining the primitives and chasing the core of what a
"reasoning-integrity regime" is: sharper novelty and yield estimators, the
guarantees the quadrant partition invites, and — the direction this is really aimed
at — instrumenting the internal trajectory of a model's own reasoning, where the
same two questions (is this making progress, or recurring without yield) become a
statement about computation rather than about a task.

---

## Reading the figures

Each evaluation writes several; each follows a rule that exists because the obvious
version misleads.

**Time series** — all runs from one substrate share axes, with the θ band shaded.
Per-run autoscaling makes a clean run's noise look like a stall.

**Telemetry trajectory** — completeness × PE × SI in 3-D. Colour is time; dots grow
with consecutive no-gain occupancy, so a stall looks like a pile-up.

**Regime portrait** — gain × SI × PE. A stall and a finished-idle process both sit
at zero gain and are separated only by SI and PE together.

**Evidence geometry** — where evidence actually went, one shared PCA basis and
viewport. Read the caption's variance number: high (kinematic, 93%) trust fine
positions; low (BOIR, 56%) read only the spread.

**Phase portrait & absorption plane** — the two views above; each point one
exchange, coloured by regime / quadrant.

---

## What's honest about this repo

- Synthetic **task data**, not synthetic embeddings — the interview uses a real model.
- The core equations are unchanged from the research code, and a test replays every
  exchange against the original implementation to prove it.
- Stalls are labelled by scenario design, never by a rule derived from the monitor's
  own output.
- A within-run AUC is suppressed when its only clean exchanges are warm-up exchanges
  — that number would be 1.000 for free.
- Where the method has limits, they're stated: no threshold for the interview, no
  correctness guarantee for the estimator, and the two phase views diverge where
  geometry and pressure genuinely say different things.
- The optional PPO example is an integration pattern, **not** a performance claim.

## Layout

```text
telemetry_tools_geometric.py     the PE and SI equations — shared, unchanged
substrate_binding.py             B = (M, Y, S, e), the runner, the JSONL contract
calibration_manifest.json        every constant, in one place

corpus_hiker.py                  substrate 1: interview
binding_kinematic.py             substrate 2: robot arm
bayesian_intent.py               substrate 3: the estimator being watched
synthetic_intent_trajectories.py     its four scenarios, from a seed
binding_boir.py                      and its binding

exp1_evaluation.py               run substrate 1   (--hash-embeddings for no-Ollama)
exp_kinematic_evaluation.py      run substrate 2
exp_boir_evaluation.py           run substrate 3

phase_space.py                   regime classifier over the trajectory contract
phase_space_report.py            phase portrait from any substrate's trajectories
latent_absorption.py             novelty × yield quadrant decomposition
latent_absorption_report.py      quadrant plane + paired-decoy validation
plotting.py / visualisation.py   figures; the viewer reads any substrate's JSONL
tests/                           78 tests — no network, no Ollama, no MuJoCo
optional/ppo_ast_demo.py         wiring AST into an RL loop
```

## Running the interview experiment with real embeddings

```bash
ollama pull qwen3-embedding && ollama serve
python precompute_embeddings.py
python check_embedding_separation.py    # confirms the regimes still separate
python exp1_evaluation.py
```

Without Ollama, `--hash-embeddings` runs the same pipeline on deterministic hash
vectors — a plumbing check, not the experiment, and the output says so on every line.

## Scope

- SI measures **low-yield recurrence under a declared binding**. It is not a
  diagnosis of *why*.
- PE and SI are **telemetry**. What you do with them — alert, penalise, truncate,
  intervene — is a separate decision, kept deliberately downstream.
- The full empirical study is under review. This repository is the method, on
  synthetic data.

MIT licensed.
