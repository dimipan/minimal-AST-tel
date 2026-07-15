"""Optional minimal PPO integration for the synthetic SAR binding.

Install with ``pip install -e '.[rl]'``. This script demonstrates how AST can
be exposed to a controller; it is not a benchmark or a performance claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # repo root (this file lives in optional/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as exc:  # pragma: no cover - optional dependency path
    raise SystemExit("Install optional dependencies with: pip install -e '.[rl]'") from exc

from corpus_hiker import iter_all_evidence, HIKER_DIMENSIONS, make_hiker_binding
from substrate_binding import TelemetryRunner


class ObjectiveMetricsCallback(BaseCallback):
    """
    Logs purely objective evaluation metrics to TensorBoard, allowing fair
    comparisons between agents regardless of their internal reward functions.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._running_si = []
        self._running_gain = []
        self._turn_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            for info in infos:
                if "si" in info:
                    self._running_si.append(float(info["si"]))
                if "gain" in info:
                    self._running_gain.append(float(info["gain"]))
                
        self._turn_count += 1

        dones = self.locals.get("dones")
        if dones:
            for i, done in enumerate(dones):
                if done:
                    # Log Total Objective Gain
                    if self._running_gain:
                        self.logger.record("objective/total_knowledge_gain", float(np.sum(self._running_gain)))
                    
                    # Log SI metrics
                    if self._running_si:
                        self.logger.record("objective/si_penalty_mean", float(np.mean(self._running_si)))
                        self.logger.record("objective/si_penalty_total", float(np.sum(self._running_si)))
                    
                    # Log Efficiency
                    self.logger.record("objective/turns_taken", float(self._turn_count))

                    self._running_si = []
                    self._running_gain = []
                    self._turn_count = 0
                    
        return True


class SyntheticSarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, 
        *, 
        ast_observation: bool, 
        si_penalty: float = 5.0, 
        max_turns: int = 100,
        discount_factor: float = 0.6,
        tau: float = 0.8
    ):
        super().__init__()
        self.ast_observation = bool(ast_observation)
        self.si_penalty = float(si_penalty)
        self.max_turns = int(max_turns)
        self.discount_factor = float(discount_factor)
        self.tau = float(tau)
        
        self.binding = make_hiker_binding()
        self.all_evidence = list(iter_all_evidence())
        
        # --- PRE-FLIGHT SANITY CHECK (Maximum-Based Update) ---
        max_achievable = {dim: 0.0 for dim in HIKER_DIMENSIONS}
        for evidence in self.all_evidence:
            scores = self.binding.score(evidence)
            for dim, score in scores.items():
                if dim in max_achievable:
                    max_achievable[dim] = max(max_achievable[dim], score * self.discount_factor)

        unsolvable_dims = {dim: peak for dim, peak in max_achievable.items() if peak < self.tau}
        if unsolvable_dims:
            print("\n" + "!" * 60)
            print("  CRITICAL ERROR: MDP IS MATHEMATICALLY UNSOLVABLE!")
            print(f"  Because updates are MAXIMUM-based, a discount of {self.discount_factor}")
            print(f"  prevents these dimensions from ever reaching tau ({self.tau}):")
            for dim, peak in unsolvable_dims.items():
                print(f"    - '{dim}' peaks at {peak:.3f}")
            print("  Please increase --discount, decrease --tau, or expand the corpus.")
            print("!" * 60 + "\n")
            raise RuntimeError("Environment initialization aborted: Unsolvable dimensions.")

        # Action space mapped to the full list of evidence items
        self.action_space = spaces.Discrete(len(self.all_evidence))
        
        base_size = len(HIKER_DIMENSIONS) + 1
        ast_size = base_size + len(HIKER_DIMENSIONS) + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(ast_size if self.ast_observation else base_size,),
            dtype=np.float32,
        )
        self.runner: TelemetryRunner
        self.turn = 0
        self.last_pe = {dim: 0.0 for dim in HIKER_DIMENSIONS}
        self.last_si = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        first_vector = self.binding.embed(self.all_evidence[0])
        self.runner = TelemetryRunner(
            HIKER_DIMENSIONS,
            first_vector.size,
            importance=self.binding.importance,
            dependencies=self.binding.dependencies,
        )
        self.turn = 0
        self.last_pe = {dim: 0.0 for dim in HIKER_DIMENSIONS}
        self.last_si = 0.0
        return self._observation(), {}

    def _observation(self) -> np.ndarray:
        values = [self.runner.state.values[dim] for dim in HIKER_DIMENSIONS]
        turn = [self.turn / self.max_turns]
        if not self.ast_observation:
            return np.asarray([*values, *turn], dtype=np.float32)
        pe = [self.last_pe[dim] for dim in HIKER_DIMENSIONS]
        si = [self.last_si]
        return np.asarray([*values, *pe, *si, *turn], dtype=np.float32)

    def step(self, action: int):
        evidence = self.all_evidence[int(action)]
        target = evidence.target
        
        # --- Apply the discount factor to the absolute incoming scores ---
        raw_scores = self.binding.score(evidence)
        discounted_scores = {k: v * self.discount_factor for k, v in raw_scores.items()}
        
        pe, si, gains, _diag = self.runner.step(
            self.binding.embed(evidence),
            discounted_scores,
            target,
        )
        self.last_pe = pe
        self.last_si = si
        self.turn += 1
        
        gain = float(sum(gains.values()))
        if self.ast_observation:
            reward = gain - self.si_penalty * si
        else:
            reward = gain
            
        terminated = all(self.runner.state.values[d] >= self.tau for d in HIKER_DIMENSIONS)
        truncated = (self.turn >= self.max_turns) or (si >= 0.05)
        
        info = {
            "gain": gain, 
            "si": si, 
            "knowledge": self.runner.state.values.copy(),
            "evidence_id": evidence.evidence_id,
            "target": target
        }
        return self._observation(), reward, terminated, truncated, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "ast"), default="ast")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=222)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--si-penalty", type=float, default=5.0)
    parser.add_argument("--model-out", type=Path, default=Path("outputs/ppo_ast_demo"))
    args = parser.parse_args()

    env = SyntheticSarEnv(
        ast_observation=(args.variant == "ast"),
        discount_factor=args.discount,
        tau=args.tau,
        si_penalty=args.si_penalty
    )
    print("variant=", args.variant, "ast_observation=", env.ast_observation, "obs_shape=", env.observation_space.shape)
    
    # Using CPU explicitly to prevent the stable-baselines3 MlpPolicy warning
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1, tensorboard_log="./ppo_tensorboard/", device="cpu", ent_coef=0.02)
    
    # Using the ObjectiveMetricsCallback
    model.learn(total_timesteps=args.steps, callback=ObjectiveMetricsCallback(verbose=1))
    
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.model_out))
    print(f"saved {args.variant} integration demo to {args.model_out}.zip")


if __name__ == "__main__":
    main()
