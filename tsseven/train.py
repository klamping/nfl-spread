import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
import joblib
import random
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Prediction Model Script")
parser.add_argument('--model', '-m', type=str, required=True, help="model to build")
args = parser.parse_args()

modelName = f"{args.model}"

# Load the dataset
file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model-data', f"seasons-no-2024-pointDiff-{modelName}.csv")
data = pd.read_csv(file_path)

# Check the data structure
# print(data.head())

# Separate features and target
X = data.drop(columns=['Target'])
y = data['Target']

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler for future use
scaler_save_path = os.path.join(os.path.dirname(__file__), 'models', f"scaler-{modelName}.joblib")
os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
joblib.dump(scaler, scaler_save_path)
# print(f'Scaler saved to {scaler_save_path}')

random_int = 7835#random.randint(0, 10000)
print(random_int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_int)

# Initialize the Random Forest model
rf_model = RandomForestRegressor(random_state=random_int)

# Early stopping setup using GradientBoostingRegressor
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_int)
best_model = None
best_val_loss = float('inf')

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    model = GradientBoostingRegressor(random_state=random_int, n_iter_no_change=10, validation_fraction=0.1)
    model.fit(X_train_fold, y_train_fold)

    val_loss = mean_squared_error(y_val_fold, model.predict(X_val_fold))
    # print(f'Validation Loss (MSE): {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = clone(model)

# Train the best model on the entire training data
best_model.fit(X_train, y_train)

# Save the model for future use
model_save_path = os.path.join(os.path.dirname(__file__), 'models', f"best_model-{modelName}.joblib")
joblib.dump(best_model, model_save_path)
# print(f'Model saved to {model_save_path}')

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Feature importance
# feature_importance = best_model.feature_importances_
# feature_importance_df = pd.DataFrame({'Feature': data.drop(columns=['Target']).columns, 'Importance': feature_importance})
# feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
# print(feature_importance_df)

# # Save feature importance to a CSV
# feature_importance_save_path = os.path.join(os.path.dirname(__file__), 'models', f"feature_importance-{modelName}.csv")
# feature_importance_df.to_csv(feature_importance_save_path, index=False)
# print(f'Feature importance saved to {feature_importance_save_path}')
