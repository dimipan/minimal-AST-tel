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

# --- CHANGED: Imported iter_all_evidence instead of EFFICIENT ---
from corpus_hiker import iter_all_evidence, HIKER_DIMENSIONS, make_hiker_binding
from substrate_binding import TelemetryRunner

class SIRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._running_si = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            for info in infos:
                si = info.get("si")
                if si is not None:
                    # record step-level SI (will appear in TB as scalar)
                    self.logger.record("env/si_step", float(si))
                    self._running_si.append(float(si))

        # when an episode ends, log episode summary stats
        dones = self.locals.get("dones")
        if dones:
            for i, done in enumerate(dones):
                if done and self._running_si:
                    self.logger.record("env/si_episode_mean", float(np.mean(self._running_si)))
                    self.logger.record("env/si_episode_max", float(np.max(self._running_si)))
                    self._running_si = []
        return True


class SyntheticSarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, ast_observation: bool, si_penalty: float = 1.0, max_turns: int = 10):
        super().__init__()
        self.ast_observation = bool(ast_observation)
        self.si_penalty = float(si_penalty)
        self.max_turns = int(max_turns)
        self.binding = make_hiker_binding()
        
        # --- CHANGED: Load all unique evidence items into a list ---
        self.all_evidence = list(iter_all_evidence())
        
        # Verify all dimensions can still be targeted
        covered_targets = {ev.target for ev in self.all_evidence}
        missing = set(HIKER_DIMENSIONS) - covered_targets
        if missing:
            raise RuntimeError(f"missing productive evidence for: {sorted(missing)}")

        # --- CHANGED: The action space is now the total number of questions, not dimensions ---
        self.action_space = spaces.Discrete(len(self.all_evidence))
        
        base_size = len(HIKER_DIMENSIONS) + 1
        ast_size = base_size + len(HIKER_DIMENSIONS) # +1
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
        # Initialize the embedder using the first item in our full evidence list
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
        return np.asarray([*values, *pe, *turn], dtype=np.float32)

    def step(self, action: int):
        # --- CHANGED: Retrieve the specific evidence item based on action ---
        evidence = self.all_evidence[int(action)]
        target = evidence.target
        
        pe, si, gains, _diag = self.runner.step(
            self.binding.embed(evidence),
            self.binding.score(evidence),
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
            
        terminated = all(self.runner.state.values[d] >= 0.8 for d in HIKER_DIMENSIONS)
        truncated = (self.turn >= self.max_turns) or (si >= 0.1)
        
        # Pass back the specific evidence ID so we can see what the agent actually asked
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
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-out", type=Path, default=Path("outputs/ppo_ast_demo"))
    args = parser.parse_args()

    env = SyntheticSarEnv(ast_observation=(args.variant == "ast"))
    print("variant=", args.variant, "ast_observation=", env.ast_observation, "obs_shape=", env.observation_space.shape)
    
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1, tensorboard_log="./ppo_tensorboard/", device="cpu", ent_coef=0.02)
    model.learn(total_timesteps=args.steps, callback=SIRecorderCallback(verbose=1))
    
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.model_out))
    print(f"saved {args.variant} integration demo to {args.model_out}.zip")


if __name__ == "__main__":
    main()



# """Optional minimal PPO integration for the synthetic SAR binding.

# Install with ``pip install -e '.[rl]'``. This script demonstrates how AST can
# be exposed to a controller; it is not a benchmark or a performance claim.
# """

# from __future__ import annotations

# import argparse
# from pathlib import Path
# import sys

# import numpy as np

# ROOT = Path(__file__).resolve().parents[1]  # repo root (this file lives in optional/)
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# try:
#     import gymnasium as gym
#     from gymnasium import spaces
#     from stable_baselines3 import PPO
#     from stable_baselines3.common.callbacks import BaseCallback
# except ImportError as exc:  # pragma: no cover - optional dependency path
#     raise SystemExit("Install optional dependencies with: pip install -e '.[rl]'") from exc

# from corpus_hiker import EFFICIENT, HIKER_DIMENSIONS, make_hiker_binding
# from substrate_binding import TelemetryRunner

# class SIRecorderCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super().__init__(verbose)
#         self._running_si = []

#     def _on_step(self) -> bool:
#         infos = self.locals.get("infos")
#         if infos:
#             for info in infos:
#                 si = info.get("si")
#                 if si is not None:
#                     # record step-level SI (will appear in TB as scalar)
#                     self.logger.record("env/si_step", float(si))
#                     self._running_si.append(float(si))

#         # when an episode ends, log episode summary stats
#         dones = self.locals.get("dones")
#         if dones:
#             for i, done in enumerate(dones):
#                 if done and self._running_si:
#                     self.logger.record("env/si_episode_mean", float(np.mean(self._running_si)))
#                     self.logger.record("env/si_episode_max", float(np.max(self._running_si)))
#                     self._running_si = []
#         return True


# class SyntheticSarEnv(gym.Env):
#     metadata = {"render_modes": []}

#     def __init__(self, *, ast_observation: bool, si_penalty: float = 1.0, max_turns: int = 50):
#         super().__init__()
#         self.ast_observation = bool(ast_observation)
#         self.si_penalty = float(si_penalty)
#         self.max_turns = int(max_turns)
#         self.binding = make_hiker_binding()
#         self.evidence_by_target = {}
#         for evidence in EFFICIENT:
#             self.evidence_by_target[evidence.target] = evidence
#         missing = set(HIKER_DIMENSIONS) - set(self.evidence_by_target)
#         if missing:
#             raise RuntimeError(f"missing productive evidence for: {sorted(missing)}")

#         self.action_space = spaces.Discrete(len(HIKER_DIMENSIONS))
#         base_size = len(HIKER_DIMENSIONS) + 1
#         ast_size = base_size + len(HIKER_DIMENSIONS) # +1
#         self.observation_space = spaces.Box(
#             low=0.0,
#             high=np.inf,
#             shape=(ast_size if self.ast_observation else base_size,),
#             dtype=np.float32,
#         )
#         self.runner: TelemetryRunner
#         self.turn = 0
#         self.last_pe = {dim: 0.0 for dim in HIKER_DIMENSIONS}
#         self.last_si = 0.0

#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
#         first_vector = self.binding.embed(next(iter(self.evidence_by_target.values())))
#         self.runner = TelemetryRunner(
#             HIKER_DIMENSIONS,
#             first_vector.size,
#             importance=self.binding.importance,
#             dependencies=self.binding.dependencies,
#         )
#         self.turn = 0
#         self.last_pe = {dim: 0.0 for dim in HIKER_DIMENSIONS}
#         self.last_si = 0.0
#         return self._observation(), {}

#     def _observation(self) -> np.ndarray:
#         values = [self.runner.state.values[dim] for dim in HIKER_DIMENSIONS]
#         turn = [self.turn / self.max_turns]
#         if not self.ast_observation:
#             return np.asarray([*values, *turn], dtype=np.float32)
#         pe = [self.last_pe[dim] for dim in HIKER_DIMENSIONS]
#         return np.asarray([*values, *pe, *turn], dtype=np.float32)

#     def step(self, action: int):
#         target = HIKER_DIMENSIONS[int(action)]
#         evidence = self.evidence_by_target[target]
#         pe, si, gains, _diag = self.runner.step(
#             self.binding.embed(evidence),
#             self.binding.score(evidence),
#             target,
#         )
#         self.last_pe = pe
#         self.last_si = si
#         self.turn += 1
#         gain = float(sum(gains.values()))
#         if self.ast_observation:
#             reward = gain - self.si_penalty * si #gain - (self.si_penalty * si)
#         else:
#             reward = gain
#         # reward = gain - (self.si_penalty * si if self.ast_observation else 0.0)
#         terminated = all(self.runner.state.values[d] >= 0.8 for d in HIKER_DIMENSIONS)
#         truncated = (self.turn >= self.max_turns) or (si >= 0.15)
#         info = {"gain": gain, "si": si, "knowledge": self.runner.state.values.copy()}
#         return self._observation(), reward, terminated, truncated, info


# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--variant", choices=("baseline", "ast"), default="ast")
#     parser.add_argument("--steps", type=int, default=30_000)
#     parser.add_argument("--seed", type=int, default=7)
#     parser.add_argument("--model-out", type=Path, default=Path("outputs/ppo_ast_demo"))
#     args = parser.parse_args()

#     env = SyntheticSarEnv(ast_observation=(args.variant == "ast"))
#     print("variant=", args.variant, "ast_observation=", env.ast_observation, "obs_shape=", env.observation_space.shape)
#     model = PPO("MlpPolicy", env, seed=args.seed, verbose=1, tensorboard_log="./ppo_tensorboard/", device="cpu", ent_coef=0.02)
#     model.learn(total_timesteps=args.steps, callback=SIRecorderCallback(verbose=1))
#     args.model_out.parent.mkdir(parents=True, exist_ok=True)
#     model.save(str(args.model_out))
#     print(f"saved {args.variant} integration demo to {args.model_out}.zip")


# if __name__ == "__main__":
#     main()
