import numpy as np
from helpers import *
from implementations import *
from helpers_analysis import *

# Set the best hyperparameters found during the fine-tuning process
best_lambda = 0.0001
best_gamma = 0.9
best_threshold = 0.393939393939394

# Load and clean the dataset
x_train, x_test, y_train, train_ids, test_ids = load_cleaned_csv_data("dataset", sub_sample=False)
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Convert y labels from {-1, 1} to binary {0, 1} for training and validation sets
y_train_binary = (y_train + 1) // 2
x_test_binary = (x_test + 1) // 2

# Initialize parameters for the model
initial_w = np.zeros(x_train.shape[1])  # Initial weights set to zeros

# Train the logistic regression model
w, loss = reg_logistic_regression(y_train_binary, X_train, best_lambda, initial_w, max_iters=1000, gamma=best_gamma)

# Compute prediction scores for the test set
y_scores = sigmoid(x_test @ w)  # Apply the sigmoid function to compute probabilities

# Generate binary predictions based on the best threshold
y_pred = np.where(y_scores >= best_threshold, 1, -1)

# Create a CSV submission file with the predictions
create_csv_submission(test_ids, y_pred, "submission.csv")