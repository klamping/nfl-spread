import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Prediction Model Script")
parser.add_argument('--date', '-d', type=str, required=True, help="Date for importing the 'model-data' lines CSV file (format: YYYY-MM-DD)")
args = parser.parse_args()

# Load the saved model
model = keras.models.load_model('nfl_spread_model-versus.keras')

# Load the saved scaler
scaler = joblib.load('scaler-versus.save')

# Load and preprocess the new data
file_path = f"../model-data/{args.date}-lines.csv"  # Update with your local file path
new_data = pd.read_csv(file_path)

matchups = new_data['Matchup']
spread = new_data['Spread']

# List all columns in the data
all_columns = new_data.columns.tolist()

# Specify non-feature columns
non_feature_columns = ['Matchup']  # Add any other non-feature columns if present

# Identify feature columns
feature_columns = [col for col in all_columns if col not in non_feature_columns]

# Extract features for prediction
X_new = new_data[feature_columns]
X_new = X_new.to_numpy()

# Scale the features
X_new_scaled = scaler.transform(X_new)

# Make predictions
predicted_probs = model.predict(X_new_scaled)
threshold = 0.5
predictions = (predicted_probs >= threshold).astype(int).flatten()

# Add predictions to the data
new_data['covered_prediction'] = predictions
new_data['prediction_probability'] = predicted_probs.flatten()

# Calculate confidence as the absolute difference from 0.5
new_data['confidence'] = abs(new_data['prediction_probability'] - 0.5)

# Create a DataFrame with original indices and confidence
confidence_df = new_data[['confidence']].copy()
confidence_df['original_index'] = confidence_df.index

# Sort by confidence in ascending order and reset index
confidence_df = confidence_df.sort_values(by='confidence', ascending=True).reset_index(drop=True)

# Assign weights starting from 1 (lowest confidence gets weight 1)
confidence_df['weight'] = range(1, len(confidence_df) + 1)

# Set index back to original indices for merging
confidence_df.set_index('original_index', inplace=True)

# Merge the weights back into new_data
new_data['weight'] = confidence_df['weight']

# Display the matchup names with predictions and weights
print(new_data[['Matchup', 'covered_prediction', 'prediction_probability', 'weight']])

# Prepare results table
results = pd.DataFrame({
    "Matchup": matchups,
    "Favorite Covers": favorite_covers,
    "Points Assigned": ranked_points,
    "Prediction": predictions,
    "Spread": spread,
    "Prediction Difference": prediction_diff,
    "Confidence (%)": (confidence / confidence.max()) * 100,
    "Combined Score": combined_score,
    "Points Assigned": ranked_points,
})

# Sort results by points assigned (most confident first)
results = results.sort_values(by="Points Assigned", ascending=False)

# Display the table
print(results)

# Optionally, save the table to a CSV
results.to_csv(f"predictions_with_combined_score_{args.date}.csv", index=False)
