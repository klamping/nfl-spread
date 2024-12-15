import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
# from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
import joblib
import shap

# Load the data from the CSV file
data = pd.read_csv('../model-data/seasons-again-versus.csv')

# Inspect and clean labels
print("Unique values in 'covered' column before cleaning:", data['Favorite Covered Spread'].unique())
data = data[data['Favorite Covered Spread'].isin([0, 1])]
print("Unique values in 'covered' column after cleaning:", data['Favorite Covered Spread'].unique())

# Ensure labels are integers
data['Favorite Covered Spread'] = data['Favorite Covered Spread'].astype(int)

# Assume 'covered' is the label column indicating if the favorite covered the spread (1) or not (0)
# Separate features and labels
X = data.drop('Favorite Covered Spread', axis=1)
y = data['Favorite Covered Spread']

# Convert to NumPy arrays
X = X.to_numpy()
y = y.to_numpy()

# Normalize the features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler to a file
joblib.dump(scaler, 'scaler-versus.save')

# Split the data into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Calculate class weights
# class_weights = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y_train),
#     y=y_train
# )
# class_weights_dict = dict(zip(np.unique(y_train), class_weights))

# print("Class Weights:", class_weights_dict)

# For the training set
unique, counts = np.unique(y_train, return_counts=True)
print("Training set class distribution:")
print(dict(zip(unique, counts)))

# For the validation set
unique, counts = np.unique(y_val, return_counts=True)
print("Validation set class distribution:")
print(dict(zip(unique, counts)))

# def create_model(optimizer='adam', neurons=64):
#     model = keras.Sequential([
#         keras.Input(shape=(X_train.shape[1],)),
#         layers.Dense(neurons, activation='relu'),
#         layers.Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=32)
# param_grid = {
#     'neurons': [32, 64, 128],
#     'optimizer': ['adam', 'sgd'],
#     'batch_size': [16, 32, 64],
#     'epochs': [50, 100, 200]
# }
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
# grid_result = grid.fit(X_train, y_train) 

# Build the model
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    # layers.Dense(64, activation='relu'),
    # layers.Dropout(0.3),
    # layers.Dense(32, activation='relu'),
    # layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model with appropriate loss function and optimizer
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Set up EarlyStopping callback to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',       # Monitor the validation loss
    patience=10,               # Stop after 5 epochs with no improvement
    restore_best_weights=True # Restore the weights from the epoch with the best value of the monitored quantity
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in skf.split(X_scaled, y):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y[train_index], y[val_index]
    model.fit(X_train, y_train, epochs=7, validation_data=(X_val, y_val))

# Train the model with early stopping
# history = model.fit(
#     X_train,
#     y_train,
#     epochs=100,               # Set a high number of epochs to allow early stopping to kick in
#     batch_size=64,
#     validation_data=(X_val, y_val),
#     callbacks=[early_stopping]
# )

# coefficients = pd.DataFrame({
#     'feature': X_train.columns,
#     'coefficient': model.coef_[0]
# })

# Sort by absolute value of coefficients
# coefficients['abs_coefficient'] = coefficients['coefficient'].abs()
# coefficients = coefficients.sort_values(by='abs_coefficient', ascending=False)

# Display the coefficients
# print(coefficients[['feature', 'coefficient']])

# Evaluate the model on the validation set
val_loss, val_accuracy, val_precision, val_recall = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')
print(f'Validation Precision: {val_precision}')
print(f'Validation Recall: {val_recall}')

# Predict probabilities on the validation set
y_val_pred_probs = model.predict(X_val)

# Convert probabilities to binary predictions (0 or 1)
thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]
for thresh in thresholds:
    y_val_pred_thresh = (y_val_pred_probs >= thresh).astype(int).flatten()
    print(f"Threshold: {thresh}")
    print(classification_report(y_val, y_val_pred_thresh))
y_val_pred = (y_val_pred_probs > 0.52).astype(int).flatten()


# Compute precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_val, y_val_pred_probs)

# Plot Precision vs. Threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], 'b--', label='Precision')  # precision[:-1] to align with thresholds
plt.plot(thresholds, recall[:-1], 'g-', label='Recall')         # recall[:-1] to align with thresholds
plt.xlabel('Threshold')
plt.ylabel('Precision / Recall')
plt.title('Precision and Recall vs. Threshold')
plt.legend()
plt.show()

# Calculate F1 scores for each threshold
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])

# # Find the threshold that maximizes F1 Score
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

print(f'Optimal Threshold: {optimal_threshold}')
print(f'Optimal F1 Score: {optimal_f1}')

# Assuming you have already computed y_val_pred
unique, counts = np.unique(y_val_pred, return_counts=True)
print("Predicted class distribution:")
print(dict(zip(unique, counts)))

# Print classification report
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

plt.hist(y_val_pred_probs, bins=50)
plt.title('Histogram of Predicted Probabilities')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.show()

unique, counts = np.unique(y_val_pred, return_counts=True)
print("Predicted class distribution:")
print(dict(zip(unique, counts)))

# Save the model
model.save('nfl_spread_model-versus.keras')
# perm_importance = PermutationImportance(model, random_state=42).fit(X_val, y_val)
# eli5.show_weights(perm_importance, feature_names=X_val.columns.tolist())
# eli5.show_weights(clf)
