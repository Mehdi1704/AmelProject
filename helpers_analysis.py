from helpers import *
import numpy as np

def load_cleaned_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train_cleaned.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test_cleaned.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids

def accuracy(y_true, y_pred):
    """
    Calculates the accuracy between true labels and predicted labels.

    Parameters:
    y_true (numpy array): True labels
    y_pred (numpy array): Predicted labels

    Returns:
    float: Accuracy score
    """
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

def f1_score(y_true, y_pred):
    """
    Calculates the F1-score between true labels and predicted labels.

    Parameters:
    y_true (numpy array): True labels (-1 or 1)
    y_pred (numpy array): Predicted labels (-1 or 1)

    Returns:
    float: F1-score
    """
    # True Positives (TP): correctly predicted positives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    # False Positives (FP): predicted positive but actually negative
    fp = np.sum((y_true == -1) & (y_pred == 1))
    # False Negatives (FN): predicted negative but actually positive
    fn = np.sum((y_true == 1) & (y_pred == -1))

    # Avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    if (precision + recall) == 0:
        return 0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Function to optimize threshold
def optimize_threshold(y_true, y_scores):
    best_threshold = None
    best_f1 = 0
    best_accuracy = 0

    # Generate a list of potential thresholds to try
    thresholds = np.linspace(np.min(y_scores), np.max(y_scores), 100)

    for threshold in thresholds:
        y_pred = np.where(y_scores >= threshold, 1, -1)
        current_f1 = f1_score(y_true, y_pred)
        current_acc = accuracy(y_true, y_pred)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
            best_accuracy = current_acc

    return best_threshold, best_f1, best_accuracy


def f1_score_logistic(y_true, y_pred):

    """
    Calculates the F1-score between true labels and predicted labels.

    Parameters:
    y_true (numpy array): True labels (0 or 1)
    y_pred (numpy array): Predicted labels (0 or 1)
    """
    # True Positives (TP)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    # False Positives (FP)
    fp = np.sum((y_true == 0) & (y_pred == 1))
    # False Negatives (FN)
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    if (precision + recall) == 0:
        return 0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def optimize_threshold_logistic(y_true, y_scores):
    best_threshold = None
    best_f1 = -1

    # Generate a list of potential thresholds to try
    thresholds = np.linspace(0, 1, 100)

    for threshold in thresholds:
        # Convert scores to binary predictions using the threshold
        y_pred = np.where(y_scores >= threshold, 1, 0)  # Use 0 and 1 labels
        current_f1 = f1_score_logistic(y_true, y_pred)  # Use your custom function
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold, best_f1
