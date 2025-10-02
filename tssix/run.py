import pandas as pd
import numpy as np
import joblib
import os

# Define variables for file suffixes
home_model_suffix = 'homeFave'
away_model_suffix = 'awayFave'

# Load the models and scalers
output_folder = './output'
home_model = joblib.load(os.path.join(output_folder, f'logistic_regression-{home_model_suffix}.pkl'))
home_scaler = joblib.load(os.path.join(output_folder, f'scaler-{home_model_suffix}.pkl'))
away_model = joblib.load(os.path.join(output_folder, f'logistic_regression-{away_model_suffix}.pkl'))
away_scaler = joblib.load(os.path.join(output_folder, f'scaler-{away_model_suffix}.pkl'))

# Load the CSV file with predictions data
input_file = '../model-data/2024-12-11-lines.csv'  # Path to the input CSV file with "Matchups" and "Is Favorite Home Team?"
data = pd.read_csv(input_file)

# Extract relevant columns
matchups = data['Matchup']  # Column for reference
is_home_favorite = data['Is Favorite Home Team']  # Column to determine model
X = data.drop(columns=['Matchup', 'Is Favorite Home Team'])  # Drop non-feature columns

# Ensure feature names are passed consistently to scalers
feature_columns = X.columns

# Prepare results container
predictions = []
probabilities = []

# Iterate through rows to select the correct model and make predictions
for i, row in X.iterrows():
    features = row.values.reshape(1, -1)
    if is_home_favorite[i] == 1:
        # Use home-favorite model and scaler
        scaled_features = home_scaler.transform(pd.DataFrame(features, columns=feature_columns))
        pred = home_model.predict(scaled_features)[0]
        prob = home_model.predict_proba(scaled_features)[0, 1]
    else:
        # Use away-favorite model and scaler
        scaled_features = away_scaler.transform(pd.DataFrame(features, columns=feature_columns))
        pred = away_model.predict(scaled_features)[0]
        prob = away_model.predict_proba(scaled_features)[0, 1]
    
    predictions.append(pred)
    probabilities.append(prob)

# Create a DataFrame to display results
results = pd.DataFrame({
    'Matchups': matchups,
    'Prediction': predictions,
    'Probability_Class_1': probabilities
})

# Save the predictions to a new CSV file
predictions_output_file = os.path.join(output_folder, 'predictions-results.csv')
results.to_csv(predictions_output_file, index=False)

# Print confirmation
print(f"Predictions saved to {predictions_output_file}")

# Display a preview of the predictions
print(results)
