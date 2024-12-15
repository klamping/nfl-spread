import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import joblib

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Prediction Model Script")
parser.add_argument('--model', '-m', type=str, required=True, help="model to build")
args = parser.parse_args()

# Load data
data_version = args.model
import os

data_path = os.path.join(os.getcwd(), f'../model-data/seasons-no-2024-{data_version}.csv')
data = pd.read_csv(data_path)
# Verify data is loaded correctly

# Separate features and target
X = data.drop('Favorite Covered Spread', axis=1)
y = data['Favorite Covered Spread']

# Handle missing values if any
if X.isnull().values.any():
    X = X.fillna(X.mean())

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=np.random.randint(0, 10000))
X_resampled, y_resampled = smote.fit_resample(X, y)

# Standardize the features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Define StratifiedKFold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize variables to store the results
all_test_reports = []
all_models = []

for train_index, test_index in skf.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled[train_index], X_resampled[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

    # Build the neural network model
    inputs = keras.Input(shape=(X_train.shape[1],))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model with advanced options to improve precision and recall
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            'accuracy'
        ]
    )

    # Define callbacks for early stopping and model checkpointing
    early_stopping = EarlyStopping(
        monitor='val_recall',
        patience=10,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), f'best_model_{data_version}_trimmed.keras'),
        monitor='val_loss',
        save_best_only=True
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Load the best saved model
    model.load_weights(os.path.join(os.getcwd(), f'best_model_{data_version}_trimmed.keras'))

    # Evaluate the model on the test set
    y_pred_prob = model(X_test, training=False)
    y_pred = np.where(y_pred_prob > 0.5, 1, 0)
    test_report = classification_report(y_test, y_pred, output_dict=True)
    all_test_reports.append(test_report)
    all_models.append(model)

# Print classification report for each fold
for i, (train_index, test_index) in enumerate(skf.split(X_resampled, y_resampled)):
    y_pred_fold = np.where(all_models[i](X_resampled[test_index], training=False) > 0.5, 1, 0)
    print(f"Classification Report for Fold {i + 1}:")
    print(classification_report(y_resampled.iloc[test_index], y_pred_fold))

# Find the model with the best balance and good recall scores
best_score = float('-inf')
best_model_index = -1

for i, report in enumerate(all_test_reports):
    recall_0 = report['0']['recall']
    recall_1 = report['1']['recall']
    balance = 1 - abs(recall_0 - recall_1)
    avg_recall = (recall_0 + recall_1) / 2
    score = balance * avg_recall  # Factor in both balance and quality of recall
    if score > best_score:
        best_score = score
        best_model_index = i

best_model = all_models[best_model_index]

# Print classification report for the best balanced model
best_test_index = list(skf.split(X_resampled, y_resampled))[best_model_index][1]
y_pred_best = np.where(best_model(X_resampled[best_test_index], training=False) > 0.5, 1, 0)
print(f"Classification Report for Best Balanced Model (Fold {best_model_index + 1}):")
print(classification_report(y_resampled.iloc[best_test_index], y_pred_best))

# Save the scaler
scaler_path = os.path.join(os.getcwd(), f'scaler_{data_version}_trimmed.save')
joblib.dump(scaler, scaler_path)

# Save the best balanced model
best_model.save(os.path.join(os.getcwd(), f'best_balanced_model_{data_version}_trimmed.keras'))
