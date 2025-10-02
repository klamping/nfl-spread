import os
import pandas as pd
import joblib
import argparse
import json

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Prediction Model Script")
parser.add_argument('--date', '-d', type=str, required=True, help="Date for importing the 'model-data' lines CSV file (format: YYYY-MM-DD)")
args = parser.parse_args()

# Define paths
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'random_forest_model.joblib')
scaler_path = os.path.join(base_dir, 'scaler.joblib')
matchups_file_path = os.path.join(os.path.dirname(base_dir), 'model-data', f"{args.date}-lines.csv")
predictions_output_json_path = os.path.join(base_dir, 'predictions', f"raw_predictions_{args.date}_cover.json")

# Load the model and scaler
best_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print(f'Model loaded from {model_path}')
print(f'Scaler loaded from {scaler_path}')

# Load the matchups data
matchups_data = pd.read_csv(matchups_file_path)
print("Matchups data loaded:")
print(matchups_data.head())

# Extract spread information from the 'Matchup' column
matchups_data['Spread'] = matchups_data['Matchup'].str.extract(r'\(-?([0-9]*\.?[0-9]+)\)')[0].astype(float)

# Extract features (excluding the 'Matchup' column)
X_matchups = matchups_data.drop(columns=['Matchup'])

# Scale the features
X_matchups_scaled = scaler.transform(X_matchups)

# Make predictions
y_predictions = best_model.predict(X_matchups_scaled)

# Append predictions and calculate differences
matchups_data['Prediction'] = y_predictions
matchups_data['Will Cover'] = matchups_data['Prediction'] > 0.5

# Print predictions
print("Predictions:")
print(matchups_data[['Matchup', 'Will Cover', 'Prediction', 'Spread']])

# Save predictions to a JSON file
matchups_data[['Matchup', 'Will Cover', 'Prediction', 'Spread']].to_json(predictions_output_json_path, orient='records')
print(f'Predictions saved to {predictions_output_json_path}')
