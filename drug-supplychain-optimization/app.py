# dashboard.py
import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🚚 Drug Supply Chain Optimization Dashboard")

RESULTS_DIR = "results"

# --- Load Data ---
rewards_file = os.path.join(RESULTS_DIR, "rl_rewards.csv")
orders_file = os.path.join(RESULTS_DIR, "orders_for_vrp.json")
routes_file = os.path.join(RESULTS_DIR, "vehicle_routes.json")

if not os.path.exists(rewards_file):
    st.error("Results not found! Please run 'python run_pipeline.py' first.")
else:
    rewards_df = pd.read_csv(rewards_file)
    with open(orders_file, "r") as f:
        orders_data = json.load(f)
    with open(routes_file, "r") as f:
        routes_data = json.load(f)

    # --- Display Results ---
    st.header("1. Reinforcement Learning for Inventory Management")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RL Agent Rewards")
        st.line_chart(rewards_df)
        st.info(f"Average Reward: **{rewards_df['reward'].mean():.2f}**")

    with col2:
        st.subheader("Generated Customer Orders")
        orders_df = pd.DataFrame(orders_data)
        st.dataframe(orders_df)

    st.header("2. Vehicle Routing Optimization")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Calculated Optimal Routes")
        for route_info in routes_data:
            st.write(f"**Vehicle {route_info['vehicle_id']}**: {' -> '.join(map(str, route_info['route']))}")
    
    with col4:
        st.subheader("Route Visualization")
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot depot
        ax.scatter(0, 0, c='red', s=100, label='Depot (0)', marker='s')

        # Plot customer locations
        customer_x = [order['x'] for order in orders_data]
        customer_y = [order['y'] for order in orders_data]
        ax.scatter(customer_x, customer_y, c='blue', s=50, label='Customers')

        # ✅ NEW: Define a list of colors to cycle through
        colors = ['green', 'purple', 'orange', 'cyan', 'magenta', 'brown']

        # ✅ MODIFIED: Loop with enumerate to get an index for colors
        for i, route_info in enumerate(routes_data):
            route_points = route_info['route']
            route_x = [0 if p == 0 else orders_data[p-1]['x'] for p in route_points]
            route_y = [0 if p == 0 else orders_data[p-1]['y'] for p in route_points]
            
            # ✅ NEW: Select a color and add a specific label for each vehicle
            color = colors[i % len(colors)] # Cycle through colors
            ax.plot(route_x, route_y, marker='o', linestyle='--', color=color, label=f"Vehicle {route_info['vehicle_id']}")
        
        ax.legend()
        ax.set_title("Delivery Routes")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        st.pyplot(fig)