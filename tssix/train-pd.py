import pandas as pd
import os
import ydf


# Load the data
data_path = "../model-data/seasons-no-2024.csv"
data = pd.read_csv(data_path)

# Check the structure of the data
# print(data.head())

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save data splits to temporary CSVs
train_data_path = "./data/train_data.csv"
test_data_path = "./data/test_data.csv"
train_data.to_csv(train_data_path, index=False)
test_data.to_csv(test_data_path, index=False)

# Train the model using TensorFlow Decision Forests
model = ydf.GradientBoostedTreesLearner(label="Target",task=ydf.Task.CLASSIFICATION,num_trees=500).train(train_data)

model.describe()

# Evaluate the model
evaluation = model.evaluate(test_data)
print("Evaluation:", evaluation)

# Generate predictions
model.predict(test_data)

# Analyse a model (e.g. partial dependence plot, variable importance)
model.analyze(test_data)

# Benchmark the inference speed of a model
model.benchmark(test_data)

# Save the model
model.save("./data/favorite_won_by_model")
