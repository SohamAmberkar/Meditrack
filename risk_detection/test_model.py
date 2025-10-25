import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

print("--- Final Model Evaluation Script (with Cyclic Features) ---")

# --- Step 1: Load Models and Test Data ---
print("\nStep 1: Loading saved models and preparing test data...")

try:
    classifier = joblib.load('xgboost_classifier.joblib')
    regressor = joblib.load('xgboost_regressor.joblib')
except FileNotFoundError:
    print("Error: Model files not found.")
    print("Please run the latest 'advanced_model_trainer.py' script first.")
    exit()

try:
    df = pd.read_csv('Suppy_Chain_Shipment_Data.csv', encoding='latin1')
except FileNotFoundError:
    print("Error: 'Suppy_Chain_Shipment_Data.csv' not found.")
    exit()

df['delivery recorded date'] = pd.to_datetime(df['delivery recorded date'], errors='coerce', dayfirst=True)
df.dropna(subset=['delivery recorded date', 'line item quantity', 'molecule/test type'], inplace=True)
drug_to_forecast = 'Lamivudine/Nevirapine/Zidovudine'
df_drug = df[df['molecule/test type'] == drug_to_forecast].copy()
df_timeseries = df_drug.set_index('delivery recorded date')['line item quantity'].resample('W').sum().reset_index()
df_timeseries.rename(columns={'delivery recorded date': 'ds', 'line item quantity': 'y'}, inplace=True)

# --- REPLICATE FEATURE ENGINEERING EXACTLY ---
print("Step 1b: Engineering all features for test data...")
# Lag and Rolling Average Features
df_timeseries['lag_1'] = df_timeseries['y'].shift(1)
df_timeseries['lag_2'] = df_timeseries['y'].shift(2)
df_timeseries['lag_4'] = df_timeseries['y'].shift(4)
df_timeseries['rolling_mean_4'] = df_timeseries['y'].shift(1).rolling(window=4).mean()
df_timeseries['rolling_mean_12'] = df_timeseries['y'].shift(1).rolling(window=12).mean()

# Cyclic Time Features
df_timeseries['month_sin'] = np.sin(2 * np.pi * df_timeseries['ds'].dt.month / 12)
df_timeseries['month_cos'] = np.cos(2 * np.pi * df_timeseries['ds'].dt.month / 12)
df_timeseries['week_sin'] = np.sin(2 * np.pi * df_timeseries['ds'].dt.isocalendar().week / 52)
df_timeseries['week_cos'] = np.cos(2 * np.pi * df_timeseries['ds'].dt.isocalendar().week / 52)

df_timeseries.dropna(inplace=True)

test_data = df_timeseries[df_timeseries['ds'].dt.year >= 2014].copy()

# --- FIX: Use the exact same feature list as the training script ---
# This was the source of the error. The lists must be identical.
features = [
    'lag_1', 'lag_4', 'rolling_mean_4',
    'month_sin', 'month_cos', 'week_sin', 'week_cos'
]
X_test = test_data[features]
y_test_original = test_data['y']

print("Models and test data loaded successfully.")

# --- Step 2: Make and Combine Predictions ---
print("\nStep 2: Evaluating model performance on the test set...")
prob_demand_occurs = classifier.predict_proba(X_test)[:, 1]
log_demand_amount = regressor.predict(X_test)
predicted_demand_amount = np.expm1(log_demand_amount)
predicted_demand_amount[predicted_demand_amount < 0] = 0
final_predictions = prob_demand_occurs * predicted_demand_amount
print("Step 2b: Combined predictions from both models.")

# --- Step 3: Evaluate Performance ---
mae = mean_absolute_error(y_test_original, final_predictions)
rmse = np.sqrt(mean_squared_error(y_test_original, final_predictions))

print(f"\n--- Performance Metrics (Final Model) ---")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")

avg_demand = y_test_original.mean()
if avg_demand > 0:
    accuracy_percentage = (1 - (mae / avg_demand)) * 100
    print(f"Average Weekly Demand in Test Set: {avg_demand:,.2f}")
    print(f"Model Accuracy (1 - MAE / Avg Demand): {accuracy_percentage:.2f}%")

# --- Step 4: Visualize and Sample Forecast ---
print("\nStep 4: Generating plot and sample forecast...")
test_data['predictions'] = final_predictions
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(test_data['ds'], test_data['y'], label='Actual Demand', color='dodgerblue')
ax.plot(test_data['ds'], test_data['predictions'], label='Predicted Demand', color='darkorange', linestyle='--')
ax.set_title(f'Final Model Forecast vs. Actuals', fontsize=16)
ax.set_xlabel('Date'); ax.set_ylabel('Weekly Quantity')
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig('prediction_vs_actuals_final_model.png')
print("Plot saved as 'prediction_vs_actuals_final_model.png'.")
plt.show()

print(f"\n--- Sample Forecast ---")
print(f"Predicted demand for the week of {test_data.iloc[0]['ds'].date()}: {int(final_predictions[0]):,}")
print(f"The actual demand for that week was: {int(y_test_original.iloc[0]):,}")

