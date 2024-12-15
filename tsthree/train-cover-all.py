import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# Define the model names and file paths
models_info = {
    'below-3point5': {
        'data_path': "../model-data/seasons-no-tie-below-3point5.csv",
        'model_save_path': "cover_model-below-3point5.keras",
        'scaler_save_path': "scaler-cover-below-3point5.pkl"
    },
    'middle': {
        'data_path': "../model-data/seasons-no-tie-middle.csv",
        'model_save_path': "cover_model-middle.keras",
        'scaler_save_path': "scaler-cover-middle.pkl"
    },
    'plus-7': {
        'data_path': "../model-data/seasons-no-tie-plus-7.csv",
        'model_save_path': "cover_model-plus-7.keras",
        'scaler_save_path': "scaler-cover-plus-7.pkl"
    }
}

# Loop through each model configuration
for model_name, model_info in models_info.items():
    print(f"Building and training model: {model_name}")
    
    # Load the dataset
    file_path = model_info['data_path']
    data = pd.read_csv(file_path)

    # Separate features (X) and target variable (y)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Initialize the scaler
    scaler = StandardScaler()

    # Set up KFold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Store fold metrics for evaluation
    fold_metrics = []

    # Perform cross-validation
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Training Fold {fold + 1}...")
        
        # Split data into training and validation sets
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Scale the features
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Build the model
        model = Sequential([
            tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # Add early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        # Train the model
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            callbacks=[early_stopping],
            epochs=200,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate the model on the validation set
        loss, accuracy, precision, recall = model.evaluate(X_val_scaled, y_val, verbose=0)
        print(f"Fold {fold + 1} - Validation Loss: {loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
        fold_metrics.append((loss, accuracy, precision, recall))

        # Calculate confusion matrix and classification report
        y_val_pred = (model.predict(X_val_scaled) > 0.5).astype(int)
        cm = confusion_matrix(y_val, y_val_pred)
        print(f"Fold {fold + 1} - Confusion Matrix:\n{cm}")
        print(f"Fold {fold + 1} - Classification Report:\n{classification_report(y_val, y_val_pred)}")

    # Calculate average performance across folds
    average_loss = np.mean([metric[0] for metric in fold_metrics])
    average_accuracy = np.mean([metric[1] for metric in fold_metrics])
    average_precision = np.mean([metric[2] for metric in fold_metrics])
    average_recall = np.mean([metric[3] for metric in fold_metrics])

    print(f"{model_name} - Average Loss Across Folds: {average_loss}")
    print(f"{model_name} - Average Accuracy Across Folds: {average_accuracy}")
    print(f"{model_name} - Average Precision Across Folds: {average_precision}")
    print(f"{model_name} - Average Recall Across Folds: {average_recall}")

    # Save the final model (trained on the last fold for demonstration)
    model.save(model_info['model_save_path'])
    print(f"Final model saved as '{model_info['model_save_path']}'")

    joblib.dump(scaler, model_info['scaler_save_path'])
