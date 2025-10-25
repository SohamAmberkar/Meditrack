import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Step 2: Training the Inventory Risk Classification Model ---")

try:
    df = pd.read_csv('drug_inventory_levels.csv')
    print("Inventory data 'drug_inventory_levels.csv' loaded.")
except FileNotFoundError:
    print("Error: 'drug_inventory_levels.csv' not found.")
    print("Please run the 'generate_inventory_data.py' script first.")
    exit()

# --- Prepare Data for Modeling ---
# Define the features the model will use to make its decision
features = [
    'average_weekly_demand',
    'demand_volatility',
    'current_stock_level',
    'demand_forecast',
    'forecast_to_stock_ratio',
    'stock_coverage_weeks'
]
target = 'risk_level'

X = df[features]
y = df[target]

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

# --- Train the Random Forest Model ---
print("\nTraining a Random Forest Classifier...")
# Initialize the model. We use class_weight='balanced' to help the model
# perform well even if there are more 'Low' risk items than 'High' risk.
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the model on the training data
model.fit(X_train, y_train)
print("Model training complete.")

# --- Evaluate the Model ---
print("\n--- Model Evaluation ---")
# Make predictions on the test data (data the model has never seen)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Save the Trained Model ---
joblib.dump(model, 'risk_classifier_model.joblib')
print("\n--- SUCCESS ---")
print("Final model saved as 'risk_classifier_model.joblib'.")

# --- Example of How to Use the Model ---
print("\n--- Example Prediction ---")
# Create a sample of a new drug's data
new_drug_data = pd.DataFrame([{
    'average_weekly_demand': 50000,
    'demand_volatility': 120000, # High volatility
    'current_stock_level': 40000,
    'demand_forecast': 60000,
    'forecast_to_stock_ratio': 60000 / 40001, # High stockout risk
    'stock_coverage_weeks': 40000 / 60001
}])
predicted_risk = model.predict(new_drug_data)
print(f"Data for a new drug: {new_drug_data.to_dict('records')[0]}")
print(f"--> Predicted Risk Level: '{predicted_risk[0]}'")

# --- [NEW] Generate Accuracy Comparison Graph ---
print("\n--- Generating Accuracy Comparison Graph ---")
try:
    # Count the occurrences of each risk level in the actual test data
    actual_counts = y_test.value_counts().reset_index()
    actual_counts.columns = ['Risk Level', 'Count']
    actual_counts['Type'] = 'Actual Data'

    # Count the occurrences of each risk level in the model's predictions
    predicted_counts = pd.Series(y_pred).value_counts().reset_index()
    predicted_counts.columns = ['Risk Level', 'Count']
    predicted_counts['Type'] = 'Model Prediction'

    # Combine into a single DataFrame for plotting
    plot_data = pd.concat([actual_counts, predicted_counts])

    # Create the bar graph
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_data, x='Risk Level', y='Count', hue='Type', order=['Low', 'Medium', 'High'])
    
    plt.title('Actual vs. Predicted Risk Level Counts (Model Accuracy)', fontsize=16)
    plt.ylabel('Number of Drugs in Test Set')
    plt.xlabel('Risk Level')
    plt.legend(title='Data Source')
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('risk_model_accuracy_comparison.png')
    print("--> Graph saved as 'risk_model_accuracy_comparison.png'")

except ImportError:
    print("\nWarning: Could not generate graph. Please install 'matplotlib' and 'seaborn' libraries to get the plot.")
    print("Run: pip install matplotlib seaborn")
except Exception as e:
    print(f"\nAn error occurred while generating the graph: {e}")

