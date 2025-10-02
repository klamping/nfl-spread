import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import random
import joblib

# Load Data
data = pd.read_csv('../model-data/seasons-no-2024-aboveEight.csv')

# Preprocessing
X = data.drop(columns=["Target"])  # Features
y = data["Target"]  # Target column

# Ensure no non-numeric columns exist
X = X.select_dtypes(include=['number'])

# Add scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
random_state = random.randint(0, 10000)  # True randomization for the random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Model Training
rf_model = RandomForestClassifier(random_state=random_state, n_estimators=100)
rf_model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(rf_model, 'random_forest_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Model Evaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Detailed Performance Metrics
classification_report_str = classification_report(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
print(classification_report_str)
print("\nConfusion Matrix:\n")
print(confusion_matrix_result)
