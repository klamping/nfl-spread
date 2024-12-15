import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Prediction Model Script")
parser.add_argument('--date', '-d', type=str, required=True, help="Date for importing the 'model-data' lines CSV file (format: YYYY-MM-DD)")
args = parser.parse_args()

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

spread = 'threeFive'

# Load the trained model and scaler
model = keras.models.load_model(os.path.join(__location__, f"best_balanced_model_{spread}.keras"))
scaler = joblib.load(os.path.join(__location__, f"./scaler_{spread}.save"))

# Load new data
file_path = os.path.join(__location__, f"../model-data/{args.date}-lines.csv")  # Update with your local file path
new_data = pd.read_csv(file_path)

# Clean column names
new_data.columns = new_data.columns.str.strip()

# Extract 'Matchup' column as labels
matchup_labels = new_data['Matchup']
new_data = new_data.drop('Matchup', axis=1)

# Handle missing values
if new_data.isnull().values.any():
    new_data = new_data.fillna(new_data.mean())

# Ensure the columns are in the same order as during training
# Since the same column names are in the CSV, we assume they are in the same order

# Standardize the features using the same scaler
X_new = scaler.transform(new_data)

# Make predictions
predictions = model.predict(X_new).flatten()

# Calculate distance from 0.5
confidence = np.abs(predictions - 0.5)

# Determine if the favorite will cover
favorite_will_cover = predictions > 0.5

# Create a DataFrame with the results
results = pd.DataFrame({
    'Matchup': matchup_labels,
    'Favorite Covers': favorite_will_cover,
    'Prediction': predictions,
    'Confidence': confidence * 100
})

results_df = pd.DataFrame(results)

# Assign points based on confidence
results_df = results_df.sort_values(by='Confidence', ascending=False)
results_df['Points Assigned'] = np.arange(1, len(results_df) + 1)[::-1]

# Sort the DataFrame by points assigned in descending order
results_df = results_df.sort_values(by='Points Assigned', ascending=False)

# Print the DataFrame
print(results_df)

# Create the 'predictions' directory if it doesn't exist
os.makedirs("predictions", exist_ok=True)

# Save the results to a CSV file
results_df.to_csv(os.path.join(__location__, f"predictions/predictions_cover_{args.date}.csv"), index=False)