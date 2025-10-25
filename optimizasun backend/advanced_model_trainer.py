import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import GridSearchCV
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

print("--- Final Model Optimization via Hyperparameter Tuning ---")

# --- PHASE 1: Data Loading and Feature Engineering ---
print("\n--- Phase 1: Data Loading and Feature Engineering ---")
# This section remains the same as it correctly prepares the data.
print("Step 1: Loading and cleaning data...")
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

print("Step 2: Engineering all features (lag, rolling, cyclic)...")
df_timeseries['lag_1'] = df_timeseries['y'].shift(1)
df_timeseries['lag_4'] = df_timeseries['y'].shift(4)
df_timeseries['rolling_mean_4'] = df_timeseries['y'].shift(1).rolling(window=4).mean()
df_timeseries['month_sin'] = np.sin(2 * np.pi * df_timeseries['ds'].dt.month / 12)
df_timeseries['month_cos'] = np.cos(2 * np.pi * df_timeseries['ds'].dt.month / 12)
df_timeseries['week_sin'] = np.sin(2 * np.pi * df_timeseries['ds'].dt.isocalendar().week / 52)
df_timeseries['week_cos'] = np.cos(2 * np.pi * df_timeseries['ds'].dt.isocalendar().week / 52)
df_timeseries.dropna(inplace=True)

df_timeseries['demand_occurred'] = (df_timeseries['y'] > 0).astype(int)
df_timeseries['y_log'] = np.log1p(df_timeseries[df_timeseries['y'] > 0]['y'])
print("Step 3: Created separate targets for classification and regression.")
print("-" * 50)

# --- PHASE 2: Splitting Data ---
print("\n--- Phase 2: Splitting Data for Training ---")
train_data = df_timeseries[df_timeseries['ds'].dt.year < 2014]
features = [
    'lag_1', 'lag_4', 'rolling_mean_4',
    'month_sin', 'month_cos', 'week_sin', 'week_cos'
]
X_train = train_data[features]
print(f"Training data contains {len(X_train)} weeks.")

# --- PHASE 3: Hyperparameter Tuning and Final Model Training ---
print("\n--- Phase 3: Finding the Best Model via Hyperparameter Tuning ---")
print("This step may take several minutes to complete...")

# --- Part 1: Tune and Train the Classifier ---
print("\nStep 3a: Tuning the demand occurrence classifier...")
y_train_classifier = train_data['demand_occurred']
# Define a parameter grid to search
param_grid_classifier = {
    'max_depth': [3, 5],
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.05]
}
classifier_grid = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False),
    param_grid=param_grid_classifier,
    cv=3, # 3-fold cross-validation
    scoring='accuracy',
    n_jobs=-1, # Use all CPU cores
    verbose=1 # Show progress
)
classifier_grid.fit(X_train, y_train_classifier)
best_classifier = classifier_grid.best_estimator_
print(f"Best classifier parameters found: {classifier_grid.best_params_}")
joblib.dump(best_classifier, 'xgboost_classifier.joblib')
print("Optimized classifier model saved as 'xgboost_classifier.joblib'")

# --- Part 2: Tune and Train the Regressor ---
print("\nStep 3b: Tuning the demand amount regressor...")
train_demand_only = train_data[train_data['demand_occurred'] == 1]
X_train_regressor = train_demand_only[features]
y_train_regressor_log = train_demand_only['y_log']
# Define a parameter grid for the regressor
param_grid_regressor = {
    'max_depth': [3, 5],
    'n_estimators': [500, 1000],
    'learning_rate': [0.05, 0.01]
}
regressor_grid = GridSearchCV(
    estimator=xgb.XGBRegressor(objective='reg:squarederror'),
    param_grid=param_grid_regressor,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
regressor_grid.fit(X_train_regressor, y_train_regressor_log)
best_regressor = regressor_grid.best_estimator_
print(f"Best regressor parameters found: {regressor_grid.best_params_}")
joblib.dump(best_regressor, 'xgboost_regressor.joblib')
print("Optimized regressor model saved as 'xgboost_regressor.joblib'")

print("\n--- SUCESS ---")
print("Final, optimized two-part model has been trained and saved.")

