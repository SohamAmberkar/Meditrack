# src/rl/demo_run.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
# ✅ FIX 1: Use absolute import starting from src
from env import SupplyChainEnv 

# Define the results directory
RESULTS_DIR = "results"
MODEL_PATH = os.path.join(RESULTS_DIR, "ppo_supplychain")
RECORDS_PATH = os.path.join(RESULTS_DIR, "simulation_results.csv")
PLOT_PATH = os.path.join(RESULTS_DIR, "simulation_plots.png")

# Load environment and model
env = SupplyChainEnv()
# ✅ FIX 2: Load the model from the correct path inside the 'results' folder
model = PPO.load(MODEL_PATH, env=env) 

# Run simulation and collect results
obs, info = env.reset()
records = []

# Increased steps for a better visualization
for step in range(1, 101): 
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action.item())
    done = terminated or truncated
    
    record = {
        "Step": step,
        "Inventory": info["inventory"],
        "Demand": info["demand"],
        # ✅ FIX 3: Record the integer action, not the array
        "Action": action.item(), 
        "Reward": reward
    }
    records.append(record)
    print(f"Step: {step:02d} | Action: {action.item()} | Demand: {info['demand']:2d} | Inventory: {info['inventory']:4.0f} | Reward: {reward:6.2f}")
    if done:
        obs, info = env.reset()

# Save results to CSV in the 'results' folder
df = pd.DataFrame(records)
df.to_csv(RECORDS_PATH, index=False)
print(f"\nSaved results to {RECORDS_PATH}")

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(df["Step"], df["Inventory"], marker="o")
plt.title("Inventory Levels Over Time")
plt.ylabel("Inventory")

plt.subplot(3, 1, 2)
plt.plot(df["Step"], df["Demand"], marker="x", color="orange")
plt.title("Demand Over Time")
plt.ylabel("Demand")

plt.subplot(3, 1, 3)
plt.plot(df["Step"], df["Reward"], marker="s", color="green")
plt.title("Rewards Over Time")
plt.xlabel("Step")
plt.ylabel("Reward")

plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()
print(f"Saved plot as {PLOT_PATH}")