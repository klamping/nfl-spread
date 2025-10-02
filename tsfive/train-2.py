import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers

# -----------------------------------------------------------------------
# 1. Data Loading and Preprocessing
# -----------------------------------------------------------------------

# Load datasets
data_path = os.path.join(os.getcwd(), f'../model-data/seasons-no-2024-pointDiff.csv')

data = pd.read_csv(data_path)
significant_features_df = pd.read_csv("Significant_Correlations_with_Target.csv")
significant_features = significant_features_df['Feature'].tolist()

target_column = "Target"
all_features = significant_features + [target_column]
data = data[all_features].dropna()

X = data[significant_features].values
y = data[target_column].values

# First, separate out a final test set (hold-out)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

# -----------------------------------------------------------------------
# 2. Model Building Function
# -----------------------------------------------------------------------

def build_model(input_dim):
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.4),
        layers.Dense(1, activation='linear')
    ])
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=0.001, weight_decay=1e-5),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    return model


# -----------------------------------------------------------------------
# 3. K-Fold Cross-Validation
# -----------------------------------------------------------------------
k = 5  # number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

fold_mse_scores = []
fold_mae_scores = []

for train_index, val_index in kf.split(X_train_full):
    X_train_fold, X_val_fold = X_train_full[train_index], X_train_full[val_index]
    y_train_fold, y_val_fold = y_train_full[train_index], y_train_full[val_index]

    # Build a new model for this fold
    model = build_model(input_dim=X_train_full.shape[1])
    
    # Early stopping for each fold
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping_cb],
        verbose=0
    )

    # Evaluate on this fold's validation set
    val_loss, val_mae = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    fold_mse_scores.append(val_loss)
    fold_mae_scores.append(val_mae)

# Print out cross-validation results
cv_mse_mean = np.mean(fold_mse_scores)
cv_mse_std = np.std(fold_mse_scores)
cv_mae_mean = np.mean(fold_mae_scores)
cv_mae_std = np.std(fold_mae_scores)

print(f"Cross-Validation MSE: {cv_mse_mean:.4f} (+/- {cv_mse_std:.4f})")
print(f"Cross-Validation MAE: {cv_mae_mean:.4f} (+/- {cv_mae_std:.4f})")

# -----------------------------------------------------------------------
# 4. Final Model Training on Entire Training Set and Test Evaluation
# -----------------------------------------------------------------------
# After you're satisfied with CV performance, you can train a final model on all of X_train_full.
final_model = build_model(input_dim=X_train_full.shape[1])

early_stopping_cb = callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True
)

history = final_model.fit(
    X_train_full, y_train_full,
    validation_split=0.15,
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping_cb],
    verbose=1
)

test_loss, test_mae = final_model.evaluate(X_test, y_test, verbose=0)
test_rmse = np.sqrt(test_loss)
print(f"Final Test MSE: {test_loss:.4f}")
print(f"Final Test MAE: {test_mae:.4f}")
print(f"Final Test RMSE: {test_rmse:.4f}")

import matplotlib.pyplot as plt

# Extract loss and validation loss from the history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
