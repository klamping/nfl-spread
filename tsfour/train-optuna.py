import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import optuna
from optuna.integration import TFKerasPruningCallback
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
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, 'scaler.save')

# Define the Optuna objective function
def objective(trial):
    # Split the training data for validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )
    
    # Suggest hyperparameters
    n_layers = trial.suggest_int('n_layers', 1, 3)
    activation = trial.suggest_categorical('activation', ['relu', 'elu', 'leaky_relu'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.6, step=0.1)
    l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Build the model
    model = keras.Sequential()
    model.add(layers.InputLayer(shape=(X_train.shape[1],)))
    
    for i in range(n_layers):
        num_units = trial.suggest_int(f'n_units_l{i}', 32, 256, step=32)
        if activation == 'leaky_relu':
            model.add(layers.Dense(num_units))
            model.add(layers.LeakyReLU())
        else:
            model.add(layers.Dense(num_units, activation=activation))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        if l2_reg > 0:
            model.add(layers.Dense(num_units, activation=activation, kernel_regularizer=keras.regularizers.l2(l2_reg)))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            'accuracy'
        ]
    )
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Include Optuna's pruning callback
    pruning_callback = TFKerasPruningCallback(trial, monitor='val_loss')
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=batch_size,
        validation_data=(X_valid, y_valid),
        callbacks=[early_stopping, pruning_callback],
        verbose=0
    )
    
    # Evaluate the model on the validation set
    score = model.evaluate(X_valid, y_valid, verbose=0)
    val_loss = score[0]  # Loss is the first element
    
    return val_loss

# Create an Optuna study and optimize
study = optuna.create_study(direction='minimize', study_name='keras_optuna_study')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters
print('Best hyperparameters:')
for key, value in study.best_params.items():
    print(f'{key}: {value}')

# Train the final model with the best hyperparameters
best_params = study.best_params

# Rebuild the model with best hyperparameters
n_layers = best_params['n_layers']
activation = best_params['activation']
dropout_rate = best_params['dropout_rate']
l2_reg = best_params['l2_reg']
learning_rate = best_params['learning_rate']
batch_size = best_params['batch_size']

model = keras.Sequential()
model.add(layers.InputLayer(shape=(X_train_full.shape[1],)))

for i in range(n_layers):
    num_units = best_params[f'n_units_l{i}']
    if activation == 'leaky_relu':
        model.add(layers.Dense(num_units))
        model.add(layers.LeakyReLU())
    else:
        model.add(layers.Dense(num_units, activation=activation))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    if l2_reg > 0:
        model.add(layers.Dense(num_units, activation=activation, kernel_regularizer=keras.regularizers.l2(l2_reg)))

model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
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
    patience=10,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_loss',
    save_best_only=True
)

# Train the final model
history = model.fit(
    X_train_full,
    y_train_full,
    epochs=100,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint]
)

# Load the best saved model
model.load_weights('best_model.keras')

# Evaluate the model on the test set
test_loss, test_precision, test_recall, test_accuracy = model.evaluate(X_test, y_test)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Generate classification report
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# Save the final model
model.save('final_model.keras')
