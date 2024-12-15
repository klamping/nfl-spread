import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Prediction Model Script")
parser.add_argument('--date', '-d', type=str, required=True, help="Date for importing the 'model-data' lines CSV file (format: YYYY-MM-DD)")
args = parser.parse_args()

# Get the current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved model
model_file_path = os.path.join(current_dir, "nfl_point_diff_model.keras")
model = tf.keras.models.load_model(model_file_path)

# Load the scaler
scaler_file_path = os.path.join(current_dir, "scaler.joblib")
scaler = joblib.load(scaler_file_path)

# Load the feature selection metadata
selected_features_path = os.path.join(current_dir, "selected_features.csv")
correlation_features = pd.read_csv(selected_features_path)
selected_features = correlation_features['Feature'].tolist()

# Load the game data to predict
prediction_data_path = os.path.join(current_dir, "..", "model-data", f"{args.date}-lines.csv")
prediction_data = pd.read_csv(prediction_data_path)

# Extract the matchup labels
matchups = prediction_data['Matchup']

# Ensure only selected features are used
X_pred = prediction_data['Spread']

# Scale the data using the saved scaler
X_pred_scaled = scaler.transform(X_pred)

# Make predictions
predictions = model.predict(X_pred_scaled)
predicted_point_differentials = predictions.flatten()

# Add predictions to the dataset
prediction_data['Predicted_Point_Differential'] = predicted_point_differentials

# Save the predictions to a new CSV file
output_file_path = os.path.join(current_dir, "predicted_game_outcomes.csv")
prediction_data.to_csv(output_file_path, index=False)

# Output matchups with predictions
for matchup, point_diff in zip(matchups, predicted_point_differentials):
    print(f"Matchup: {matchup}, Predicted Point Differential: {point_diff}")

print(f"Predicted point differentials saved to {output_file_path}")
