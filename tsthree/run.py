import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Prediction Model Script")
parser.add_argument('--date', '-d', type=str, required=True, help="Date for importing the 'model-data' lines CSV file (format: YYYY-MM-DD)")
args = parser.parse_args()

# Load the prediction dataset
file_path = f"../model-data/{args.date}-lines-pointDiff.csv"  # Update with your local file path
data = pd.read_csv(file_path)

correlation_file_path = "./features/all.csv"
new_correlation_data = pd.read_csv(correlation_file_path)

# Clean up column names and feature values by stripping whitespace
new_correlation_data.columns = new_correlation_data.columns.str.strip()
new_correlation_data['Feature'] = new_correlation_data['Feature'].str.strip()

# List of columns to keep
features_to_keep = new_correlation_data['Feature'].tolist()

# Filter the dataset to include only the specified features
filtered_data = data[[col for col in features_to_keep if col in data.columns]]

# Separate features (X) and target variable (y)
X = filtered_data.values
# Extract features and matchups
matchups = data['Matchup']
spread = data['Spread']

# Load the scaler used during training
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X)  # Apply the same scaling

# Load the trained model
model = tf.keras.models.load_model("model-pointDiff.keras")

# Predict target values and confidence scores
predictions = model.predict(X_new_scaled).flatten()

# Calculate confidence as the inverse of prediction variance (if available)
# Here, assume confidence is absolute value of predictions for simplicity
confidence = np.abs(predictions)

# Calculate the difference between prediction and spread
prediction_diff = predictions - spread

# Determine if the favorite covers the spread
favorite_covers = prediction_diff > 0

# Combine confidence and prediction difference to assign points
# combined_score = confidence + (prediction_diff.clip(lower=0) / spread) # 682
# combined_score = confidence * np.abs(prediction_diff) # 684
# combined_score = confidence # 678
combined_score = confidence # 688
# combined_score = 0.6 * confidence + 0.4 * np.abs(prediction_diff) # 683
# combined_score = 0.4 * confidence + 0.6 * np.abs(prediction_diff) # 683
# combined_score = (0.2 * confidence) * np.abs(prediction_diff) # 684
# combined_score = confidence * np.exp(np.abs(prediction_diff))

# Rank predictions by combined score and assign points (1-13)
ranked_indices = np.argsort(-combined_score)  # Sort descending by combined score
points = np.arange(len(data), 0, -1)  # Points from 13 to 1
ranked_points = points[np.argsort(ranked_indices)]  # Assign points to rankings

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

# Create the 'predictions' directory if it doesn't exist
os.makedirs("predictions", exist_ok=True)

# Optionally, save the table to a CSV
results.to_csv(f"predictions/predictions_point_diff_{args.date}.csv", index=False)
