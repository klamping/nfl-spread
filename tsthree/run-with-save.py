import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import argparse
import joblib

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Prediction Model Script")
parser.add_argument('--date', '-d', type=str, required=True, help="Date for importing the 'model-data' lines CSV file (format: YYYY-MM-DD)")
args = parser.parse_args()

# Load the prediction dataset
file_path = f"../model-data/{args.date}-lines.csv"  # Update with your local file path
data = pd.read_csv(file_path)

correlation_file_path = "../Trimmed_Significant_Features.csv"
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
scaler = joblib.load("scaler_with_kfold.pkl")
X_new_scaled = scaler.transform(X)  # Apply the same scaling

# Load the trained model
model = tf.keras.models.load_model("final_model_with_kfold.keras")

# Function to make predictions with dropout enabled
def predict_with_dropout(model, X, n_iter=100):
    predictions = []
    for _ in range(n_iter):
        dropout_model_output = model(X, training=True)
        predictions.append(dropout_model_output.numpy())
    predictions = np.array(predictions)
    return predictions.mean(axis=0), predictions.std(axis=0)

# Get mean prediction and standard deviation (used as confidence)
mean_prediction, prediction_std = predict_with_dropout(model, X_new_scaled)
confidence = 1 / (1 + prediction_std)

# Calculate the difference between prediction and spread
prediction_diff = mean_prediction - spread

# Determine if the favorite covers the spread
favorite_covers = prediction_diff > 0

# Combine confidence and prediction difference to assign points
combined_score = confidence * np.abs(prediction_diff)

# Rank predictions by combined score and assign points (1-13)
ranked_indices = np.argsort(-combined_score)  # Sort descending by combined score
points = np.arange(len(data), 0, -1)  # Points from 13 to 1
ranked_points = points[np.argsort(ranked_indices)]  # Assign points to rankings

# Prepare results table
results = pd.DataFrame({
    "Matchup": matchups,
    "Favorite Covers": favorite_covers,
    "Points Assigned": ranked_points,
    "Prediction": mean_prediction,
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
