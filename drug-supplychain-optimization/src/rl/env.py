# src/env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class SupplyChainEnv(gym.Env):
    """
    An advanced supply chain environment with dynamic ordering and lead times.
    """
    def __init__(self):
        super(SupplyChainEnv, self).__init__()

        # --- NEW: Action space with dynamic order quantities ---
        self.action_space = spaces.Discrete(4)  # Actions: 0, 1, 2, 3
        self.action_to_quantity = {0: 0, 1: 10, 2: 20, 3: 40}

        # --- NEW: Added lead time ---
        self.lead_time = 5
        
        # --- NEW: Expanded observation space ---
        # It now includes: [inventory, demand] + stock in transit for each day of the lead time
        observation_size = 2 + self.lead_time
        self.observation_space = spaces.Box(low=0, high=1000, shape=(observation_size,), dtype=np.float32)

        self.max_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.inventory = 50
        self.current_step = 0

        # --- NEW: Keep track of orders in transit ---
        self.in_transit_inventory = deque([0] * self.lead_time, maxlen=self.lead_time)

        self.demand_over_time = np.random.randint(10, 31, size=self.max_steps)
        
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        # --- NEW: Stock from the oldest order arrives ---
        arriving_stock = self.in_transit_inventory.popleft()
        self.inventory += arriving_stock
        
        # --- NEW: Place a new order into the transit pipeline ---
        order_quantity = self.action_to_quantity[action]
        self.in_transit_inventory.append(order_quantity)

        # Current demand for this step
        current_demand = self.demand_over_time[self.current_step]

        # Fulfill demand
        units_sold = min(self.inventory, current_demand)
        self.inventory -= units_sold

        # The reward function remains the same, but the problem is harder
        profit = units_sold * 10
        holding_cost = self.inventory * 2.0
        stockout_penalty = (current_demand - units_sold) * 15
        reward = profit - holding_cost - stockout_penalty

        # Move to the next step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        demand_index = min(self.current_step, self.max_steps - 1)
        # --- NEW: Observation now includes the in-transit queue ---
        obs_list = [self.inventory, self.demand_over_time[demand_index]] + list(self.in_transit_inventory)
        return np.array(obs_list, dtype=np.float32)

    def _get_info(self):
        demand_index = min(self.current_step, self.max_steps - 1)
        return {
            "inventory": self.inventory,
            "demand": self.demand_over_time[demand_index],
            "in_transit": list(self.in_transit_inventory),
            "customer_x": np.random.randint(1, 100),
            "customer_y": np.random.randint(1, 100)
        }