# src/train_rl.py
import os
import json
import pandas as pd
from stable_baselines3 import PPO
from src.rl.env import SupplyChainEnv

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_and_generate_orders():
    print("--- Starting RL Model Training ---")
    env = SupplyChainEnv()
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=50000)
    model.save(os.path.join(RESULTS_DIR, "ppo_supplychain"))
    print("✅ RL model trained and saved.")

    print("\n--- Generating Customer Orders from RL Simulation ---")
    orders_data = []
    rewards_log = []
    obs, info = env.reset()
    for step in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action.item())
        
        # Only log an order if there was demand
        if info["demand"] > 0:
            # ✅ FIX: Convert all NumPy numbers to standard Python integers using int()
            orders_data.append({
                "customer_id": int(step),
                "x": int(info["customer_x"]),
                "y": int(info["customer_y"]),
                "demand": int(info["demand"])
            })
        rewards_log.append(reward)

        if terminated or truncated:
            break
            
    # Save the generated orders for the VRP solver
    with open(os.path.join(RESULTS_DIR, "orders_for_vrp.json"), "w") as f:
        json.dump(orders_data, f, indent=4)
    print(f"✅ Generated and saved {len(orders_data)} customer orders.")

    # Save rewards for analysis
    pd.Series(rewards_log).to_csv(os.path.join(RESULTS_DIR, "rl_rewards.csv"), index=False, header=["reward"])
    print("✅ RL simulation rewards saved.")

if __name__ == "__main__":
    train_and_generate_orders()