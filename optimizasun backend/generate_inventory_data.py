import pandas as pd
import numpy as np

print("--- Step 1: Generating a Rich Inventory Dataset ---")

try:
    df = pd.read_csv('Suppy_Chain_Shipment_Data.csv', encoding='latin1')
    print("Source data 'Suppy_Chain_Shipment_Data.csv' loaded.")
except FileNotFoundError:
    print("Error: 'Suppy_Chain_Shipment_Data.csv' not found.")
    exit()

# Clean up column names for easier access
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')

# --- Feature Engineering at the SKU (Drug) Level ---
print("Calculating inventory metrics for each drug...")
# Group by the specific drug to analyze each one individually
inventory_df = df.groupby('molecule_test_type').agg(
    # Calculate the average weekly demand
    average_weekly_demand=('line_item_quantity', lambda x: x.sum() / df['delivery_recorded_date'].nunique()),
    # Calculate demand volatility (standard deviation of non-zero shipments)
    demand_volatility=('line_item_quantity', lambda x: x[x > 0].std())
).reset_index()

# Replace NaN volatility (for drugs with only one shipment) with 0
inventory_df['demand_volatility'].fillna(0, inplace=True)

# --- Simulate Current Stock and Forecast ---
# In a real system, this would come from a live database and your forecast model.
# Here, we simulate it to create a realistic training dataset.
# Use average demand as a proxy for the forecast
inventory_df['demand_forecast'] = inventory_df['average_weekly_demand']
# Simulate a random current stock level for each drug
np.random.seed(42) # for reproducible results
inventory_df['current_stock_level'] = inventory_df['demand_forecast'] * np.random.uniform(0.5, 5, size=len(inventory_df))
inventory_df['current_stock_level'] = inventory_df['current_stock_level'].astype(int)

# --- Engineer Risk Ratios ---
# These ratios are the core features our model will learn from.
# Understock Risk: How big is the forecast relative to what we have? High ratio = High risk
inventory_df['forecast_to_stock_ratio'] = inventory_df['demand_forecast'] / (inventory_df['current_stock_level'] + 1) # +1 to avoid division by zero
# Overstock Risk: How much extra stock do we have compared to the forecast?
inventory_df['stock_coverage_weeks'] = inventory_df['current_stock_level'] / (inventory_df['demand_forecast'] + 1)

print("Engineered risk-based features.")

# --- Define Risk Levels (Creating our Target Variable) ---
# These rules classify each drug into a risk category. This is what the model will learn to predict.
def assign_risk_level(row):
    # High Risk: High volatility, or high chance of stockout, or very high overstock
    if row['demand_volatility'] > 100000 or row['forecast_to_stock_ratio'] > 0.8 or row['stock_coverage_weeks'] > 26:
        return 'High'
    # Medium Risk: Moderate volatility or moderate stock imbalances
    elif row['demand_volatility'] > 50000 or row['forecast_to_stock_ratio'] > 0.5 or row['stock_coverage_weeks'] > 12:
        return 'Medium'
    # Low Risk: Everything else
    else:
        return 'Low'

inventory_df['risk_level'] = inventory_df.apply(assign_risk_level, axis=1)
print("Assigned risk levels to create a labeled dataset.")

# --- Save the Final Dataset ---
inventory_df.to_csv('drug_inventory_levels.csv', index=False)
print("\n--- SUCCESS ---")
print("New dataset 'drug_inventory_levels.csv' has been created successfully.")
print("This file is now ready to be used for training the risk model.")
