import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import argparse
import os
import json


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Prediction Model Script")
parser.add_argument('--date', '-d', type=str, required=True, help="Date for importing the 'model-data' lines CSV file (format: YYYY-MM-DD)")
args = parser.parse_args()

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Define spread categories
spread_categories = {
#   "a": [0.5, 1.5],
#   "oneTwoThree": [2.5],
#   "twoThreeFour": [3.5],
#   "four": [4.5],
#   "fourFiveSix": [5.5, 6.5],
#   "sevenEightNine": [7.5, 8.5, 9.5],
#   "aboveNine": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5]
#   "lower": [0.5, 1, 1.5, 2, 2.5, 3, 3.5],
#   "middle": [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5],
#   "upper": [9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5]
  "all": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5]
}

def get_spread_category(value):
    for category, ranges in spread_categories.items():
        if value in ranges:
            return category
    raise ValueError(f"Unknown spread value: {value}")

def load_models_and_scalers():
    models_and_scalers = {}
    for category in spread_categories.keys():
        model_path = os.path.join(__location__, f"models/best_balanced_model_{category}_diffs.keras")
        scaler_path = os.path.join(__location__, f"models/scaler_{category}_diffs.save")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            models_and_scalers[category] = {
                "model": keras.models.load_model(model_path),
                "scaler": joblib.load(scaler_path)
            }
        else:
            raise FileNotFoundError(f"Model or scaler for '{category}' not found.")
    return models_and_scalers

# Load all models and scalers upfront
models_and_scalers = load_models_and_scalers()

# Load new data
file_path = os.path.join(__location__, f"../model-data/{args.date}-lines.csv")  # Update with your local file path
new_data = pd.read_csv(file_path)

# Clean column names
new_data.columns = new_data.columns.str.strip()

# Extract 'Matchup' column as labels
matchup_labels = new_data['Matchup']
spread_values = new_data['Spread']
new_data = new_data.drop(['Matchup'], axis=1)

# Handle missing values
if new_data.isnull().values.any():
    new_data = new_data.fillna(new_data.mean())

# Prepare results DataFrame
results = []

for index, spread_value in enumerate(spread_values):
    try:
        category = get_spread_category(spread_value)
    except ValueError as e:
        print(e)
        continue
    
    # Fetch preloaded model and scaler
    if category not in models_and_scalers:
        raise FileNotFoundError(f"Missing model/scaler for category: {category}")

    model = models_and_scalers[category]["model"]
    scaler = models_and_scalers[category]["scaler"]

    # Standardize the features using the scaler
    X_sample = new_data.iloc[[index]]
    X_sample_scaled = scaler.transform(X_sample)

    # Make predictions
    prediction = model.predict(X_sample_scaled).flatten()[0]
    spread_str = str(spread_value)

    # Append to results
    results.append({
        'Matchup': matchup_labels.iloc[index],
        'Prediction': prediction,
        'Spread': spread_str
    })

# Create the 'predictions' directory if it doesn't exist
os.makedirs("predictions", exist_ok=True)

# Convert results to a JSON-serializable format
serializable_results = [
    {key: (float(value) if isinstance(value, (np.float32, np.float64)) else value)
     for key, value in result.items()}
    for result in results
]

# Save the results to a JSON file
json_output_path = os.path.join(__location__, f"predictions/raw_predictions_{args.date}.json")
with open(json_output_path, 'w') as json_file:
    json.dump(serializable_results, json_file, indent=4)

print(f"Results saved to {json_output_path}")