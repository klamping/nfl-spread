import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

# Load data
data = pd.read_csv('../model-data/seasons-no-2024.csv')

# Separate features and target
X = data.drop('Favorite Covered Spread', axis=1)
y = data['Favorite Covered Spread']

# Handle missing values if any
if X.isnull().values.any():
    X = X.fillna(X.mean())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, 'scaler.save')

# Build the neural network model with tuned hyperparameters
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),  # Add Input layer
    layers.Dense(
        32,  # n_units_l0
        activation='elu',  # activation
        kernel_regularizer=regularizers.l2(7.233254333537566e-05)  # l2_reg
    ),
    layers.BatchNormalization(),
    layers.Dropout(0.6),  # dropout_rate
    layers.Dense(1, activation='sigmoid')  # Output layer
])

# Compile the model with tuned learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0022804795676220767),  # learning_rate
    loss='binary_crossentropy',
    metrics=[
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        'accuracy'
    ]
)

# Define callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_loss',
    save_best_only=True
)

# Train the model with tuned batch size
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,  # batch_size
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1  # Set to 1 to see training progress
)

# Load the best saved model
model.load_weights('best_model.keras')

# Evaluate the model on the test set
test_loss, test_precision, test_recall, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Generate classification report
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# Save the final model for future predictions
model.save('final_model.keras')
