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
model_paths = {
    "two": os.path.join(base_dir, 'models', 'best_model-two.joblib'),
    "three": os.path.join(base_dir, 'models', 'best_model-three.joblib'),
    "four": os.path.join(base_dir, 'models', 'best_model-four.joblib'),
    "twoThreeFour": os.path.join(base_dir, 'models', 'best_model-twoThreeFour.joblib'),
    "fiveSix": os.path.join(base_dir, 'models', 'best_model-fiveSix.joblib'),
    "fourFiveSix": os.path.join(base_dir, 'models', 'best_model-fourFiveSix.joblib'),
    "seven": os.path.join(base_dir, 'models', 'best_model-seven.joblib'),
    "eight": os.path.join(base_dir, 'models', 'best_model-eight.joblib'),
    "aboveEight": os.path.join(base_dir, 'models', 'best_model-aboveEight.joblib'),
    "lower": os.path.join(base_dir, 'models', 'best_model-lower.joblib'),
    "middle": os.path.join(base_dir, 'models', 'best_model-middle.joblib'),
    "middleUp": os.path.join(base_dir, 'models', 'best_model-middleUp.joblib'),
    "all": os.path.join(base_dir, 'models', 'best_model-all.joblib'),
}
scaler_paths = {
    "two": os.path.join(base_dir, 'models', 'scaler-two.joblib'),
    "three": os.path.join(base_dir, 'models', 'scaler-three.joblib'),
    "four": os.path.join(base_dir, 'models', 'scaler-four.joblib'),
    "twoThreeFour": os.path.join(base_dir, 'models', 'scaler-twoThreeFour.joblib'),
    "fiveSix": os.path.join(base_dir, 'models', 'scaler-fiveSix.joblib'),
    "fourFiveSix": os.path.join(base_dir, 'models', 'scaler-fourFiveSix.joblib'),
    "seven": os.path.join(base_dir, 'models', 'scaler-seven.joblib'),
    "eight": os.path.join(base_dir, 'models', 'scaler-eight.joblib'),
    "aboveEight": os.path.join(base_dir, 'models', 'scaler-aboveEight.joblib'),
    "lower": os.path.join(base_dir, 'models', 'scaler-lower.joblib'),
    "middle": os.path.join(base_dir, 'models', 'scaler-middle.joblib'),
    "middleUp": os.path.join(base_dir, 'models', 'scaler-middleUp.joblib'),
    "all": os.path.join(base_dir, 'models', 'scaler-all.joblib'),
}
matchups_file_path = os.path.join(os.path.dirname(base_dir), 'model-data', f"{args.date}-lines-pointDiff.csv")
predictions_output_json_path = os.path.join(base_dir, 'predictions', f"raw_predictions_{args.date}.json")

# Load the matchups data
matchups_data = pd.read_csv(matchups_file_path)
# print("Matchups data loaded:")
# print(matchups_data.head())

# Extract spread information from the 'Matchup' column
matchups_data['Spread'] = matchups_data['Matchup'].str.extract(r'\(-?([0-9]*\.?[0-9]+)\)')[0].astype(float)

# Determine which model and scaler to use based on Spread
def select_model_and_scaler(spread):
    if spread in [0.5, 2.5, 3.5]:
        return model_paths['lower'], scaler_paths['lower']
    elif spread in [4.5]:
        return model_paths['middle'], scaler_paths['middle']
    elif spread in [5.5]:
        return model_paths['all'], scaler_paths['all']
    elif spread in [6.5, 8.5]:
        return model_paths['middleUp'], scaler_paths['middleUp']
    elif spread in [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5]:
        return model_paths['aboveEight'], scaler_paths['aboveEight']
    else:
        return model_paths['all'], scaler_paths['all']

# Process each rw to select the appropriate model and scaler and predict
def predict_for_row(row):
    model_path, scaler_path = select_model_and_scaler(row['Spread'])
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(pd.DataFrame([row.drop(['Matchup', 'Spread'])]))
    return model.predict(X_scaled)[0]

# Extract features (excluding the 'Matchup' and 'Spread' columns)
X_matchups = matchups_data.drop(columns=['Matchup', 'Spread'])

# Apply predictions
matchups_data['Prediction'] = matchups_data.apply(predict_for_row, axis=1)
matchups_data['Difference'] = matchups_data['Prediction'] - matchups_data['Spread']
matchups_data['Will Cover'] = matchups_data['Difference'] > 0

# Print predictions
print("Predictions:")
print(matchups_data[['Matchup', 'Will Cover', 'Prediction', 'Spread', 'Difference']])

# Save predictions to a JSON file
matchups_data[['Matchup', 'Will Cover', 'Prediction', 'Spread', 'Difference']].to_json(predictions_output_json_path, orient='records')
print(f'Predictions saved to {predictions_output_json_path}')
