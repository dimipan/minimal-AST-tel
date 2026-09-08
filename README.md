# Acquisition-State Telemetry

You watch a process acquire information and, at each step, measure two simple things: how
much of the task is still unresolved, and whether new evidence is just repeating itself
without adding anything. That is the whole method.

- **PE (progress)** — how much of the task is left to resolve.
- **SI (friction)** — whether incoming evidence is recurring while nothing gets resolved.

The point is that the same two measurements work on any process. The equations do not know
whether they are watching a conversation, a robot, or another algorithm. One frozen monitor
is bound to three deliberately different toy processes and asked the same question of each.

Worked example — a witness interview. The task is to establish 8 facts about a missing
hiker: `location, time, clothing, direction, injuries, companions, gear, mood`.

```
turn  witness says                        resolves        PE (left)   SI (friction)
1     "red jacket, near the north ridge"   clothing, loc.  7.0 open    0.02   new info
2     "he was heading up, toward the peak"  direction       5.5 open    0.03   new info
3     "around 2 or 3 in the afternoon"      time            4.7 open    0.02   new info
4     "like I said, red jacket"             —               4.7 open    0.31   ← repeat, no gain
5     "the red one, definitely red"         —               4.7 open    0.44   ← still circling
6     "he had a small pack with him"        gear            4.1 open    0.05   new info again
```

PE drifts down as facts get filled and tells you how much task is left. SI stays near zero
while answers land, jumps at turns 4–5 when the same clothing evidence returns with no gain,
then drops again at turn 6. If the witness repeated "red jacket" *after* clothing was already
fully resolved, SI would stay quiet — that is finished, not stuck. The monitor never looks
inside the witness; it only reads what the binding reports.

Everything here is small and synthetic: no GPU, no simulator, runs in under a minute,
reproduces from a clean clone.

## How a process plugs in

Each process declares a **binding** — four things. Everything after the binding (the two
measurements, the constants) is shared and identical across all three processes.

```text
B = (M, Y, S, e)

M  what counts as done       the facts/categories to resolve
Y  what evidence looks like   an utterance, a robot frame, a posterior window
S  how to score it            evidence -> how resolved each item is, 0..1
e  how to represent it        evidence -> a vector
```

## The three substrates, and what each shows

**Interview (language).** The example above, with real `qwen3-embedding` vectors. An
efficient run barely registers friction; both a re-asking interviewer and a witness who has
run dry score clearly higher, and the two stalls rank apart cleanly.

**Robot arm (physical).** Reach, grasp, lift, place, in pure NumPy. Two runs matter most: an
arm that circles the object without grasping fires loudly (SI ~0.40), while an arm that
finishes the task and then hovers stays silent (SI ~0.04). Both are "doing nothing new," but
only the unfinished one is a stall. That gap is the whole idea — repetition after the job is
done is not friction.

**Estimator (inference).** Not a task. A recursive estimator guessing which of four goals an
operator wants, with AST watching the estimator itself. It cleanly flags the run where two
goals stay tied, and — importantly — stays quiet on a run where the estimator resolves fast
but lands on the *wrong* goal. SI is right to stay quiet, because the estimator is not stuck.
It is confident and wrong, which is a different failure.

> AST reports whether a process is still learning. It says nothing about whether what it
> learned is true.

## Two things to know before trusting a number

- **SI's ranking is reliable; its exact value is not.** It shifts with the embedding space, so
  a threshold tuned on one representation does not transfer. Thresholds live in
  `per_binding_calibration`, not the shared block. The robot and estimator set one; the
  interview does not (sessions are too short to place it honestly) and reports ranking instead.
- **The scorer must be able to fall.** AST records the best score each item ever reached, which
  is fine when progress cannot be undone — but a Bayesian posterior can un-learn. So the
  estimator's scorer measures how *settled* a hypothesis is, not how *close to true*, because a
  lucky swing toward the truth would otherwise lock in and hide a real stall.

## Two views over the trajectory

Every substrate writes the same trajectory format, so both views run on any of them.

**Phase space** sorts each step into `productive`, `churn` (spinning, task still open),
`converged` (done), or `quiet-incomplete` (starved, not spinning). `churn` and `converged`
are both "nothing resolved," so a plain repetition detector cannot tell them apart; friction
and pressure together can. The verdict is a simple count, the churn fraction.

**Latent absorption** splits friction into novelty × yield. It separates a process that is
looping on old ground from one throwing out fresh evidence that still resolves nothing. On a
robot, being stuck looks like sitting still; in language, it can look like fluent motion
going nowhere — and this view tells those apart.

```bash
python phase_space_report.py       outputs/kinematic/*.jsonl
python latent_absorption_report.py outputs/boir/*.jsonl
```

## Layout

```text
telemetry_tools_geometric.py     the two measurements (PE and SI), shared and unchanged
substrate_binding.py             B = (M, Y, S, e), the runner, the trajectory format
calibration_manifest.json        every constant, in one place

corpus_hiker.py                  substrate 1: interview
binding_kinematic.py             substrate 2: robot arm
bayesian_intent.py               substrate 3: the estimator being watched
synthetic_intent_trajectories.py   its four scenarios
binding_boir.py                    its binding

exp1_evaluation.py               run substrate 1 (--hash-embeddings to skip Ollama)
exp_kinematic_evaluation.py      run substrate 2
exp_boir_evaluation.py           run substrate 3

phase_space.py                   the regime classifier
phase_space_report.py            phase view from any substrate
latent_absorption.py             the novelty × yield split
latent_absorption_report.py      absorption view from any substrate
plotting.py / visualisation.py   figures; read any substrate's trajectories
tests/                           78 tests, no network, no Ollama, no MuJoCo
optional/ppo_ast_demo.py         wiring AST into an RL loop
```

## Run it

```bash
pip install -e .
python exp_kinematic_evaluation.py
python exp_boir_evaluation.py
python -m unittest discover -s tests

# interview with real embeddings
ollama pull qwen3-embedding && ollama serve
python precompute_embeddings.py
python exp1_evaluation.py            # or --hash-embeddings to skip Ollama
```

## Notes

- Task data is synthetic; the interview embeddings are real.
- Stalls are labelled by scenario design, never by the monitor's own output.
- PE and SI are just measurements. Acting on them — alert, penalise, stop — is a separate
  decision, kept downstream; the PPO script is an integration example, not a benchmark.
- The full empirical study is under review. This repo is the method, on synthetic data.

MIT licensed.
