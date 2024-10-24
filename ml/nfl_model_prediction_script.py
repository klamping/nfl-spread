
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
data_path = 'YOUR_DATA_PATH_HERE.csv'
nfl_data = pd.read_csv(data_path)

# Step 1: Feature Selection (Using the top 25 correlated features)
nfl_data_cleaned = nfl_data.drop(columns=["Favorite Covered Spread"])

# Calculate correlations and select top 25 features
correlation_matrix = nfl_data.corr()
target_correlation = correlation_matrix["Favorite Covered Spread"].abs().sort_values(ascending=False)
top_25_features = target_correlation.index[1:26]

# Create the dataset with the top features
X = nfl_data_cleaned[top_25_features]
y = nfl_data["Favorite Covered Spread"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, 'nfl_model_logistic_regression.joblib')
joblib.dump(scaler, 'nfl_scaler.joblib')

# Make predictions with a confidence threshold
y_pred_prob = model.predict_proba(X_test_scaled)
y_pred_confidence = y_pred_prob[:, 1]
confidence_threshold = 0.7
weighted_predictions = (y_pred_confidence >= confidence_threshold).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, weighted_predictions)
report = classification_report(y_test, weighted_predictions)
print(f"Accuracy: {accuracy}")
print(report)

# Random game selection and weighting
random_indices = random.sample(range(len(X_test)), 16)
X_test_random = X_test_scaled[random_indices]
y_test_random = y_test.iloc[random_indices]
y_pred_prob_random = model.predict_proba(X_test_random)
y_pred_random = model.predict(X_test_random)
y_pred_confidence_random = y_pred_prob_random[:, 1]

# Rank the games based on confidence
confidence_order = np.argsort(y_pred_confidence_random)[::-1]
weights = list(range(16, 0, -1))

# Create table of results
results = []
for i, index in enumerate(confidence_order):
    prediction = y_pred_random[index]
    confidence = y_pred_confidence_random[index]
    actual = y_test_random.iloc[index]
    weight = weights[i]
    correct = int(prediction == actual)
    results.append((prediction, confidence, weight, correct))

# Calculate total score
total_score = sum([r[2] for r in results if r[3] == 1])
print(f"Total weighted score: {total_score}")

results_df = pd.DataFrame(results, columns=["Prediction", "Confidence", "Weight", "Correct"])
print(results_df)
