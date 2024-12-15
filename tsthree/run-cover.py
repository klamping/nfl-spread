import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import joblib
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Prediction Model Script")
parser.add_argument('--date', '-d', type=str, required=True, help="Date for importing the 'model-data' lines CSV file (format: YYYY-MM-DD)")
args = parser.parse_args()

# Load the saved model
model = tf.keras.models.load_model("cover_model.keras")

# Load the saved scaler
scaler = joblib.load("scaler-cover.pkl")

# Load the new dataset for predictions
file_path = f"../model-data/{args.date}-lines.csv"  # Update with your local file path
new_data = pd.read_csv(file_path)

# Load correlation data from CSV (used to determine which features to keep)
correlation_file_path = "./features/all.csv"
new_correlation_data = pd.read_csv(correlation_file_path)

# Clean up column names and feature values by stripping whitespace
new_correlation_data.columns = new_correlation_data.columns.str.strip()
new_correlation_data['Feature'] = new_correlation_data['Feature'].str.strip()

# List of columns to keep
features_to_keep = new_correlation_data['Feature'].tolist()

# Filter the dataset to include only the specified features
filtered_new_data = new_data[[col for col in features_to_keep if col in new_data.columns]]

# Scale the features
correlation_weights = new_correlation_data.set_index('Feature')['Correlation'].reindex(filtered_new_data.columns).fillna(1).values
weight_adjustment_factor = 0.5  # Use the same weight adjustment factor as during training
adjusted_weights = 1 + weight_adjustment_factor * (correlation_weights - 1)
X_new_weighted = filtered_new_data.values * adjusted_weights
X_new_scaled = scaler.transform(X_new_weighted)

# Make predictions
predictions = model.predict(X_new_scaled).flatten()

# Calculate confidence as the absolute value of the prediction minus 0.5
confidence = np.abs(predictions - 0.5)

# Assign points based on confidence
sorted_indices = np.argsort(confidence)[::-1]  # Sort confidence in descending order
points_assigned = np.arange(1, len(predictions) + 1)[::-1]  # Assign points from highest to lowest

# Determine if the favorite will cover
favorite_will_cover = predictions > 0.5

# Create a DataFrame for the results
matchups = new_data['Matchup'].values
results_df = pd.DataFrame({
    'Matchup': matchups,
    'Favorite Covers': favorite_will_cover,
    'Points Assigned': points_assigned[np.argsort(sorted_indices)],
    'Prediction': predictions,
    'Confidence': confidence
})

# Sort the DataFrame by points assigned in descending order
results_df = results_df.sort_values(by='Points Assigned', ascending=False)

# Print the DataFrame
print(results_df)

# Create the 'predictions' directory if it doesn't exist
os.makedirs("predictions", exist_ok=True)

# Save the results to a CSV file in the 'predictions' directory
results_df.to_csv(f"predictions/predictions_cover_{args.date}.csv", index=False)
