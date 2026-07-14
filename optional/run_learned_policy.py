import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from ppo_ast_demo import SyntheticSarEnv, HIKER_DIMENSIONS

# 1. Setup Env and Model
env = SyntheticSarEnv(ast_observation=True)
model = PPO.load("outputs/ppo_ast_demo.zip", env=env, device="cpu")

# 2. Tracking variables
obs, _ = env.reset()
knowledge_history = []
step_rewards = []

print("\n=== STARTING INFERENCE EPISODE ===")

# 3. Run Inference for ONE episode
for step in range(env.max_turns):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # --- THE FIX: Extract ordered values from the dictionary ---
    current_knowledge = [info["knowledge"][dim] for dim in HIKER_DIMENSIONS]
    knowledge_history.append(current_knowledge)
    step_rewards.append(reward)
    
    # Print what the agent actually did this turn
    target_chosen = HIKER_DIMENSIONS[int(action)]
    print(f"Turn {step+1:02d} | Action (Target): {target_chosen:<15} | Reward: {reward:+.3f} | SI Penalty: {info['si']:.3f} | Gain: {info['gain']:.3f}")
    
    if terminated or truncated:
        print("=== EPISODE FINISHED ===")
        print(f"Reason: {'Terminated (Success)' if terminated else 'Truncated (Max Turns Reached)'}")
        print(f"Total Turns: {step + 1}")
        print(f"Total Reward: {sum(step_rewards):.2f}\n")
        break

# 4. Plotting the results
knowledge_history = np.array(knowledge_history)

plt.figure(figsize=(10, 6))
plt.title("Agent Knowledge Accumulation per Turn")

# Plot each dimension's progress
for i, dim in enumerate(HIKER_DIMENSIONS):
    plt.plot(knowledge_history[:, i], label=f"{dim}", marker="o", markersize=4)

plt.axhline(y=0.75, color='r', linestyle='--', alpha=0.5, label="Goal Threshold (0.75)")

plt.xlabel("Turn")
plt.ylabel("Knowledge Value")
plt.ylim(0, 1.0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()