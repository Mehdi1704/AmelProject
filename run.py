import numpy as np
from helpers import *
from implementations import *
from helpers_analysis import *

# Load and clean the dataset
x_train, x_test, y_train, train_ids, test_ids = load_cleaned_csv_data("dataset", sub_sample=False)

# Split the training data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Convert y labels from {-1, 1} to binary {0, 1} for training and validation sets
y_train_binary = (y_train + 1) // 2
y_test_binary = (y_test + 1) // 2

# Initialize parameters for the model
best_gamma = 0.95                      # Learning rate for the logistic regression model
initial_w = np.zeros(X_train.shape[1])  # Initial weights set to zeros

# Train the logistic regression model
w, loss = logistic_regression(y_train_binary, X_train, initial_w, max_iters=1000, gamma=best_gamma)

# Compute prediction scores for the test set
y_scores = sigmoid(x_test @ w)  # Apply the sigmoid function to compute probabilities

# Optimize the threshold to maximize F1 score and accuracy
best_threshold, best_f1, best_accuracy = optimize_threshold(y_test_binary, y_scores)

# Generate binary predictions based on the best threshold
y_pred = np.where(y_scores >= best_threshold, 1, -1)

# Output the performance results
print("Threshold:", best_threshold)
print("Final Accuracy:", best_accuracy)
print("Final F1 Score:", best_f1)

# Create a CSV submission file with the predictions
create_csv_submission(test_ids, y_pred, "submission.csv")