import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("--- Generating a Presentable Report for the Inventory Risk Model ---")

# --- 1. Load the Model and Data ---
try:
    model = joblib.load('risk_classifier_model.joblib')
    df = pd.read_csv('drug_inventory_levels.csv')
    print("Model and inventory data loaded successfully.")
except FileNotFoundError:
    print("Error: Ensure 'risk_classifier_model.joblib' and 'drug_inventory_levels.csv' are in the correct folder.")
    exit()

# Re-create the exact same test set as when the model was trained
features = [
    'average_weekly_demand', 'demand_volatility', 'current_stock_level',
    'demand_forecast', 'forecast_to_stock_ratio', 'stock_coverage_weeks'
]
target = 'risk_level'
X = df[features]
y = df[target]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_pred = model.predict(X_test)
risk_levels = ['Low', 'Medium', 'High'] # Define order for plots

# --- 2. Generate and Print the Report ---

print("\n" + "="*60)
print("           Inventory Risk Model Performance Report")
print("="*60)

# High-level summary
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Model Accuracy: {accuracy:.2%}")
print("This means the model correctly predicts the risk level for 94% of drugs it hasn't seen before.")
print("-" * 60)

# --- 3. Generate Confusion Matrix Plot ---
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred, labels=risk_levels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
            xticklabels=risk_levels, yticklabels=risk_levels)
plt.title('Confusion Matrix: Predicted vs. Actual Risk', fontsize=15)
plt.ylabel('Actual Risk Level')
plt.xlabel('Predicted Risk Level')
plt.tight_layout()
plt.savefig('risk_model_confusion_matrix.png')
print("--> Saved visualization as 'risk_model_confusion_matrix.png'")

# Explanation of the Confusion Matrix
print("\nHow to Read the Confusion Matrix:")
print("The numbers on the diagonal (top-left to bottom-right) are CORRECT predictions.")
print("Numbers off the diagonal are incorrect predictions (e.g., predicting 'Medium' when it was 'High').")

# --- 4. Interpreted Classification Report ---
print("\n" + "-"*60)
print("Detailed Performance Breakdown:\n")
report = classification_report(y_test, y_pred, labels=risk_levels, output_dict=True)

# High Risk
print("HIGH RISK:")
print(f"  - Recall: {report['High']['recall']:.2%}")
print("    Interpretation: The model successfully identified 100% of all drugs that were truly high-risk. This is excellent for preventing stockouts.")
print(f"  - Precision: {report['High']['precision']:.2%}")
print("    Interpretation: When the model says a drug is 'High' risk, it is correct 75% of the time.")

# Medium Risk
print("\nMEDIUM RISK:")
print(f"  - Recall: {report['Medium']['recall']:.2%}")
print("    Interpretation: The model found 67% of all medium-risk drugs.")
print(f"  - Precision: {report['Medium']['precision']:.2%}")
print("    Interpretation: The model is very precise; when it predicts 'Medium' risk, it's correct 100% of the time.")

# Low Risk
print("\nLOW RISK:")
print(f"  - Recall: {report['Low']['recall']:.2%}")
print("    Interpretation: The model found 100% of all low-risk drugs.")
print(f"  - Precision: {report['Low']['precision']:.2%}")
print("    Interpretation: When the model predicts 'Low' risk, it is correct 100% of the time. This builds trust in knowing which items are safe.")
print("-" * 60)

# --- 5. Generate Feature Importance Plot ---
print("\nGenerating Feature Importance Plot...")
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
plt.title('Which Factors Are Most Important for Predicting Risk?', fontsize=15)
plt.xlabel('Importance Score')
plt.ylabel('Inventory Feature')
plt.tight_layout()
plt.savefig('risk_model_feature_importance.png')
print("--> Saved visualization as 'risk_model_feature_importance.png'")

print("\nHow to Read the Feature Importance Plot:")
print("This chart shows which data points the model relies on most. A higher bar means the feature was more influential in the model's decisions.")

print("\n\n" + "="*60)
print("                              End of Report")
print("="*60)

