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

# Load the new dataset for predictions
file_path = f"../model-data/{args.date}-lines.csv"  # Update with your local file path
new_data = pd.read_csv(file_path)

# Determine which model and scaler to use based on the 'Spread' column
def get_model_scaler(spread_value):
    if spread_value < 3.5:
        return "below_3.5"
    elif spread_value < 7:
        return "middle"
    else:
        return "plus_7" 

base_model_path = "cover_model-final-"
base_scaler_path = "scaler-cover-final-"

models = {
    'below_3.5': tf.keras.models.load_model(f"{base_model_path}below-3point5.keras"),
    'middle': tf.keras.models.load_model(f"{base_model_path}middle.keras"),
    'plus_7': tf.keras.models.load_model(f"{base_model_path}plus-7.keras")
}

scalers = {
    'below_3.5': joblib.load(f"{base_scaler_path}below-3point5.pkl"),
    'middle': joblib.load(f"{base_scaler_path}middle.pkl"),
    'plus_7': joblib.load(f"{base_scaler_path}plus-7.pkl")
}

# Iterate through each row, select the appropriate model, and make predictions
results = []
for index, row in new_data.iterrows():
    spread = row.loc['Spread']
    file_key = get_model_scaler(spread)

    # Load the saved model and scaler
    model = models[file_key]
    scaler = scalers[file_key]

    # Filter the dataset to include only the features present in the model
    filtered_data = row.drop('Matchup').to_frame().T

    # Scale the features
    X_scaled = scaler.transform(filtered_data.values)

    # Make prediction
    prediction = model.predict(X_scaled).flatten()[0]

    # Calculate confidence as the absolute value of the prediction minus 0.5
    confidence = np.abs(prediction - 0.5)

    # Determine if the favorite will cover
    favorite_will_cover = prediction > 0.5

    # Append the results
    results.append({
        'Matchup': row['Matchup'],
        'Favorite Covers': favorite_will_cover,
        'Prediction': prediction,
        'Confidence': confidence,
        'Spread': spread
    })

# Create a DataFrame for the results
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

# Save the results to a CSV file in the 'predictions' directory
results_df.to_csv(f"predictions/predictions_cover_{args.date}.csv", index=False)
