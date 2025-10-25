# run_pipeline.py
from src.rl.train_rl import train_and_generate_orders
from src.ilp.vrp_or_tools import solve_vehicle_routing

def main():
    print("=============================================")
    print("=== STARTING SUPPLY CHAIN & ROUTING PIPELINE ===")
    print("=============================================")
    
    # Step 1: Run RL to optimize inventory and generate orders
    train_and_generate_orders()
    
    # Step 2: Run VRP to find optimal routes for those orders
    solve_vehicle_routing()
    
    print("\n=============================================")
    print("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print("=== Check the 'results' folder for outputs. ===")
    print("=============================================")

if __name__ == "__main__":
    main()