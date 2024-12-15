import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import joblib
import argparse
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Prediction Model Script")
parser.add_argument('--date', '-d', type=str, required=True, help="Date for importing the 'model-data' lines CSV file (format: YYYY-MM-DD)")
args = parser.parse_args()

# Load the new dataset for predictions
file_path = os.path.join(__location__, f"../model-data/{args.date}-lines.csv")  # Update with your local file path
new_data = pd.read_csv(file_path)

# Determine which model, scaler, and correlation file to use based on the 'Spread' column
def get_file_mod(spread_value):
    if spread_value < 3.5:
        return "below_3.5"
    elif spread_value < 7:
        return "middle"
    else:
        return "plus_7"

models = {
    'below_3.5': tf.keras.models.load_model(os.path.join(__location__, 'cover_model-below-3point5.keras')),
    'middle': tf.keras.models.load_model(os.path.join(__location__, 'cover_model-middle.keras')),
    'plus_7': tf.keras.models.load_model(os.path.join(__location__, 'cover_model-plus-7.keras'))
}

scalers = {
    'below_3.5': joblib.load(os.path.join(__location__, 'scaler-cover-below-3point5.pkl')),
    'middle': joblib.load(os.path.join(__location__, 'scaler-cover-middle.pkl')),
    'plus_7': joblib.load(os.path.join(__location__, 'scaler-cover-plus-7.pkl'))
}

correlations = {
    'below_3.5': pd.read_csv(os.path.join(__location__, 'features', 'below-3point5.csv')),
    'middle': pd.read_csv(os.path.join(__location__, 'features', 'middle.csv')),
    'plus_7': pd.read_csv(os.path.join(__location__, 'features', 'plus-7.csv'))
}

# Iterate through each row, select the appropriate model, and make predictions
results = []
for index, row in new_data.iterrows():
    spread = row['Spread']
    file_key = get_file_mod(spread)

    # Load the saved model, scaler, and correlation data
    model = models[file_key]
    scaler = scalers[file_key]
    correlation_data = correlations[file_key]

    # Clean up column names and feature values by stripping whitespace
    correlation_data.columns = correlation_data.columns.str.strip()
    correlation_data['Feature'] = correlation_data['Feature'].str.strip()

    # List of columns to keep
    features_to_keep = correlation_data['Feature'].tolist()

    # Filter the dataset to include only the specified features
    filtered_data = row[[col for col in features_to_keep if col in new_data.columns]].to_frame().T

    # Scale the features
    correlation_weights = correlation_data.set_index('Feature')['Correlation'].reindex(filtered_data.columns).fillna(1).values
    weight_adjustment_factor = 0.5  # Use the same weight adjustment factor as during training
    adjusted_weights = 1 + weight_adjustment_factor * (correlation_weights - 1)
    X_weighted = filtered_data.values * adjusted_weights
    X_scaled = scaler.transform(X_weighted)

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
os.makedirs(os.path.join(__location__, "predictions"), exist_ok=True)

# Save the results to a CSV file in the 'predictions' directory
results_df.to_csv(os.path.join(__location__, f"predictions/predictions_cover_{args.date}.csv"), index=False)
