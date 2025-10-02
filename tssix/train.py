import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import random
import joblib

# Define variable for file suffix
file_suffix = 'homeFave'

# Load the data
file_path = f'../model-data/seasons-no-2024-{file_suffix}.csv'
data = pd.read_csv(file_path)

# Ensure output folder exists
output_folder = './output'
os.makedirs(output_folder, exist_ok=True)

# Prepare the data
X = data.iloc[:, :-1]  # Features (all columns except the last one)
y = data.iloc[:, -1]   # Target (last column)

# Scale the data to ensure feature parity
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize StratifiedKFold with a random random_state
random_state = random.randint(0, 10000)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

final_model_performance = {}

for train_index, test_index in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train a Logistic Regression model with early stopping
    model = LogisticRegression(max_iter=1000, solver='saga', warm_start=True)
    best_accuracy = 0
    patience = 5
    no_improve_counter = 0

    for iteration in range(1, 1001):
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_test)
        current_accuracy = accuracy_score(y_test, y_val_pred)

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        if no_improve_counter >= patience:
            print(f"Early stopping on iteration {iteration}")
            break

    # Final evaluation for the last fold only
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    final_model_performance = {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }

# Save the model and scaler
model_filename = os.path.join(output_folder, f'logistic_regression-{file_suffix}.pkl')
scaler_filename = os.path.join(output_folder, f'scaler-{file_suffix}.pkl')
joblib.dump(model, model_filename)
joblib.dump(scaler, scaler_filename)

# Print final results
print(f"Final Model Accuracy: {final_model_performance['accuracy']}")
print("Final Model Confusion Matrix:")
print(final_model_performance['confusion_matrix'])
print("Final Model Classification Report:")
print(final_model_performance['classification_report'])
