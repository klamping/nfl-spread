import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# Define the model names and file paths
base_data_path = "../model-data/seasons-no-2024"
base_model_save_path = "cover_model-final-"
base_scaler_save_path = "scaler-cover-final-"

models_info = {
    'below-3point5': {
        'data_path': f"{base_data_path}-below-3point5.csv",
        'model_save_path': f"{base_model_save_path}below-3point5.keras",
        'scaler_save_path': f"{base_scaler_save_path}below-3point5.pkl"
    },
    'middle': {
        'data_path': f"{base_data_path}-middle.csv",
        'model_save_path': f"{base_model_save_path}middle.keras",
        'scaler_save_path': f"{base_scaler_save_path}middle.pkl"
    },
    'plus-7': {
        'data_path': f"{base_data_path}-plus-7.csv",
        'model_save_path': f"{base_model_save_path}plus-7.keras",
        'scaler_save_path': f"{base_scaler_save_path}plus-7.pkl"
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
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Store fold metrics for evaluation
    fold_metrics = []

    # Perform cross-validation
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
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
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        # Train the model
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            callbacks=[early_stopping],
            epochs=100,
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

    # Train the final model on the entire dataset
    # X_scaled = scaler.fit_transform(X)
    # final_model = Sequential([
    #     tf.keras.layers.Input(shape=(X_scaled.shape[1],)),
    #     Dense(256, activation='relu'),
    #     Dropout(0.3),
    #     Dense(128, activation='relu'),
    #     Dropout(0.3),
    #     Dense(64, activation='relu'),
    #     Dropout(0.2),
    #     Dense(32, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])

    # final_model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #     loss='binary_crossentropy',
    #     metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    # )

    # early_stopping_final = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    # final_history = final_model.fit(
    #     X_scaled, y,
    #     epochs=60,
    #     batch_size=32,
    #     callbacks=[early_stopping_final],
    #     verbose=1
    # )

    # # Visualization of training history
    # # import matplotlib.pyplot as plt

    # # plt.figure(figsize=(12, 4))

    # # # Plot training & validation loss values
    # # plt.subplot(1, 2, 1)
    # # plt.plot(final_history.history['loss'], label='Loss')
    # # plt.title('Model Loss')
    # # plt.xlabel('Epoch')
    # # plt.ylabel('Loss')
    # # plt.legend()

    # # # Plot training accuracy values
    # # plt.subplot(1, 2, 2)
    # # plt.plot(final_history.history['accuracy'], label='Accuracy')
    # # plt.title('Model Accuracy')
    # # plt.xlabel('Epoch')
    # # plt.ylabel('Accuracy')
    # # plt.legend()

    # # plt.show()

    # # Evaluate the final model
    # final_loss, final_accuracy, final_precision, final_recall = final_model.evaluate(X_scaled, y, verbose=0)
    # print(f"{model_name} - Final Model Loss: {final_loss}")
    # print(f"{model_name} - Final Model Accuracy: {final_accuracy}")
    # print(f"{model_name} - Final Model Precision: {final_precision}")
    # print(f"{model_name} - Final Model Recall: {final_recall}")

    # # Calculate confusion matrix and classification report for the final model
    # y_pred = (final_model.predict(X_scaled) > 0.5).astype(int)
    # final_cm = confusion_matrix(y, y_pred)
    # print(f"{model_name} - Final Model Confusion Matrix:\n{final_cm}")
    # print(f"{model_name} - Final Model Classification Report:\n{classification_report(y, y_pred)}")

    # Save the final model
    model.save(model_info['model_save_path'])
    print(f"Final model saved as '{model_info['model_save_path']}'")

    joblib.dump(scaler, model_info['scaler_save_path'])
