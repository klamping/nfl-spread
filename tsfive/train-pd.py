import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import joblib

# Load the data
data_path = '../model-data/seasons-no-2024-pointDiff.csv'
data = pd.read_csv(data_path)

# Separate features and target
features = data.drop(columns=['Favorite Won By'])
target = data['Favorite Won By']

# Convert target to categories for stratification
target_bins = pd.qcut(target, q=10, labels=False)

# Convert categorical data to numerical (if any)
features = pd.get_dummies(features, drop_first=True)

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Save the scaler
scaler_path = './models/scaler.save'
joblib.dump(scaler, scaler_path)

# Learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch < 4:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1))

# Prepare StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
history_list = []
final_metrics = []

for train_index, val_index in kfold.split(features, target_bins):
    X_train, X_val = features[train_index], features[val_index]
    y_train, y_val = target.iloc[train_index], target.iloc[val_index]

    # Build the model
    model = Sequential([
        Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02), input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        Dense(1, activation='linear')  # Linear activation for regression
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=128,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    history_list.append(history.history)

    # Evaluate the model
    metrics = model.evaluate(X_val, y_val, verbose=0)
    final_metrics.append(metrics)

# Save the final model
model.save('./models/nfl_point_diff_model.keras')

# Print final model performance
avg_loss = np.mean([m[0] for m in final_metrics])
avg_mae = np.mean([m[1] for m in final_metrics])
print(f"Average Validation Loss: {avg_loss}")
print(f"Average Validation MAE: {avg_mae}")

# Plot training history of the last fold
import matplotlib.pyplot as plt
plt.plot(history_list[-1]['loss'], label='Training Loss')
plt.plot(history_list[-1]['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Training History (Last Fold)')
plt.show()
