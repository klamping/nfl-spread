import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Get the current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load data
historical_data_path = os.path.join(current_dir, "..", "model-data", "seasons-no-2024-pointDiff.csv")

# Read the dataset
historical_data = pd.read_csv(historical_data_path)

# Extract features and label
X = historical_data.drop(columns=['Favorite Won By'])
y = historical_data['Favorite Won By']

# Perform Recursive Feature Elimination with Cross-Validation (RFECV)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
scaler_file_path = os.path.join(current_dir, "scaler.joblib")
joblib.dump(scaler, scaler_file_path)

# Linear Regression for RFECV
estimator = LinearRegression()
selector = RFECV(estimator, step=1, cv=5, scoring='neg_mean_squared_error')
X_reduced = selector.fit_transform(X_scaled, y)
selected_features = [X.columns[i] for i in range(len(X.columns)) if selector.support_[i]]

print("Selected features after RFECV:", selected_features)

selected_features_path = os.path.join(current_dir, "selected_features.csv")
pd.DataFrame(selected_features, columns=["Feature"]).to_csv(selected_features_path, index=False)
print(f"Selected features saved to {selected_features_path}")

# Define the model function
def build_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

# K-fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
histories = []
evaluation_results = []

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

for train_index, test_index in kf.split(X_reduced):
    X_train, X_test = X_reduced[train_index], X_reduced[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = build_model(X_train.shape[1])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping]
    )

    eval_results = model.evaluate(X_test, y_test, verbose=1)
    evaluation_results.append(eval_results)
    histories.append(history)

# Average evaluation results
# Example evaluation
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Save the last trained model
model_file_path = os.path.join(current_dir, "nfl_point_diff_model.keras")
model.save(model_file_path)

# Ensure the last fold's history is used
history = histories[-1].history

# Plot the loss for training and validation
import matplotlib.pyplot as plt
plt.plot(history['loss'], label='Training Loss (MSE)')
plt.plot(history['val_loss'], label='Validation Loss (MSE)')

# Update axis labels and title for regression context
plt.xlabel('Epochs')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend()
plt.title('Model Training History (Last Fold)')
plt.show()
