import numpy as np
import csv

def read_header(file_path):
    """Reads the header (first row) from a CSV file."""
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    return header

def load_data(file_path):
    """Loads dataset as a numpy array, skipping the header row."""
    return np.genfromtxt(file_path, delimiter=',', skip_header=1)

def find_constant_columns(data):
    """Finds indices of columns in the data where all non-NaN values are constant."""
    constant_columns = []
    for i in range(data.shape[1]):
        unique_values = np.unique(data[:, i][~np.isnan(data[:, i])])
        if len(unique_values) == 1:
            constant_columns.append(i)
    return constant_columns

def drop_columns(data, header, columns_to_drop):
    """Drops specified columns from the data array and header list."""
    data_cleaned = np.delete(data, columns_to_drop, axis=1)
    header_cleaned = [col for i, col in enumerate(header) if i not in columns_to_drop]
    return data_cleaned, header_cleaned

def create_dummy_variables(state_column, unique_states):
    """
    Creates a dummy matrix where each unique state has its own column.
    Each row in the dummy matrix has a 1 for the matching state, 0 otherwise.
    
    Args:
        state_column (numpy array): Column with categorical values.
        unique_states (numpy array): Array of unique states.
        
    Returns:
        numpy array: A binary matrix with dummy variables for each unique state.
    """
    dummy_matrix = np.zeros((state_column.shape[0], unique_states.shape[0]))
    
    for i, state in enumerate(unique_states):
        dummy_matrix[:, i] = (state_column == state).astype(int)
    
    return dummy_matrix

def create_gen_health_dummy_variables(health_column):
    """
    Creates dummy variables for each level of general health from 1 to 5.
    Rows with values 7, 9, or NaN will have zeros in all dummy columns.
    
    Args:
        health_column (numpy array): Array of health ratings (1-5, or missing values).
        
    Returns:
        numpy array: Binary matrix with a column for each level 1 to 5.
    """
    dummy_matrix = np.zeros((health_column.shape[0], 5))
    
    for i in range(1, 6):
        dummy_matrix[:, i-1] = (health_column == i).astype(int)
    
    return dummy_matrix

def process_health_feature(x_data, x_headers, feature_name, create_dummy_func, dummy_labels):
    """
    Processes a categorical feature by creating dummy variables,
    removing the original column, and updating the headers.

    Parameters:
    - x_data: NumPy array of the dataset (training or test data)
    - x_headers: List of headers corresponding to x_data
    - feature_name: Name of the feature to process
    - create_dummy_func: Function to create dummy variables for the feature
    - dummy_labels: List of labels for the dummy variables

    Returns:
    - x_data: Updated dataset with dummy variables
    - x_headers: Updated headers list
    """
    # Find the index of the feature column
    feature_col_index = x_headers.index(feature_name)
    
    # Extract the feature column
    feature_data = x_data[:, feature_col_index]
    
    # Create dummy variables for the feature
    dummy_feature = create_dummy_func(feature_data)
    
    # Remove the original feature column
    x_data = np.delete(x_data, feature_col_index, axis=1)
    
    # Append the dummy variables to the dataset
    x_data = np.hstack((x_data, dummy_feature))
    
    # Update the headers
    x_headers = [col for col in x_headers if col != feature_name] + dummy_labels
    
    return x_data, x_headers

def replace_values(dataset, indices, replacements):
    """
    Replaces values in specified columns with new values according to a dictionary.
    
    Args:
        dataset (numpy array): Dataset in which values will be replaced.
        indices (list): List of column indices to apply replacements.
        replacements (dict): Dictionary of {old_value: new_value} pairs.
    """
    dataset_copy = dataset.copy()
    for index in indices:
        for old_value, new_value in replacements.items():
            mask = dataset_copy[:, index] == old_value
            dataset_copy[:, index][mask] = new_value
    return dataset_copy

def identify_categorical_columns(train_data, test_data, train_headers):
    """
    Identifies columns with integer values 1-9 and at most 8 unique values.
    These columns are likely categorical and suitable for dummy variable creation.
    
    Args:
        train_data (numpy array): Training dataset.
        test_data (numpy array): Testing dataset.
        train_headers (list): List of column names in the training dataset.
        
    Returns:
        list: Indices of identified categorical columns.
    """
    categorical_columns = []
    
    for idx in range(train_data.shape[1]):
        combined_column = np.concatenate((train_data[:, idx], test_data[:, idx]))
        
        # Exclude NaN values and check if values are in the range 1-9 with <=8 unique values
        unique_values = np.unique(combined_column[~np.isnan(combined_column)])
        if np.all((unique_values >= 1) & (unique_values <= 9)) and len(unique_values) <= 8:
            categorical_columns.append(idx)
    
    return categorical_columns

def create_dummy_variables_for_column(data, headers, column_idx):
    """
    Converts a specified categorical column to dummy variables, appending them to the dataset.
    Values 7, 9, and NaN are replaced with 0 (indicating no category).
    
    Args:
        data (numpy array): Dataset to update.
        headers (list): Column names of the dataset.
        column_idx (int): Index of the categorical column to convert.
        
    Returns:
        tuple: Updated data array and headers list.
    """
    column_data = data[:, column_idx]
    column_name = headers[column_idx]
    
    column_data_clean = np.where(np.isin(column_data, [7, 9]) | np.isnan(column_data), 0, column_data)
    
    data = np.delete(data, column_idx, axis=1)
    del headers[column_idx]
    
    unique_values = np.unique(column_data_clean[column_data_clean != 0])
    for val in unique_values:
        dummy_column = (column_data_clean == val).astype(int)
        data = np.column_stack((data, dummy_column))
        headers.append(f"{column_name}_{int(val)}")
    
    return data, headers

def process_categorical_columns(train_data, test_data, train_headers, test_headers):
    """
    Processes both training and testing datasets by identifying categorical columns,
    converting them to dummy variables, and ensuring consistent headers.
    
    Args:
        train_data (numpy array): Training dataset.
        test_data (numpy array): Testing dataset.
        train_headers (list): Column names for training data.
        test_headers (list): Column names for testing data.
        
    Returns:
        tuple: Cleaned training and testing datasets with updated headers.
    """
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)
    
    categorical_columns = identify_categorical_columns(train_data, test_data, train_headers)
    
    for idx in sorted(categorical_columns, reverse=True):
        train_data, train_headers = create_dummy_variables_for_column(train_data, train_headers, idx)
        test_data, test_headers = create_dummy_variables_for_column(test_data, test_headers, idx)
    
    return train_data, test_data, train_headers, test_headers

def fill_nans_with_median(train_data, test_data):
    """
    Replaces NaN values in both datasets with the median of the corresponding column
    from the training dataset.
    
    Args:
        train_data (numpy array): Training dataset.
        test_data (numpy array): Testing dataset.
        
    Returns:
        tuple: Training and testing datasets with NaNs replaced by medians.
    """
    train_data_filled = train_data.copy()
    test_data_filled = test_data.copy()
    
    for col_idx in range(train_data.shape[1]):
        median = np.nanmedian(train_data[:, col_idx])
        
        train_data_filled[np.isnan(train_data[:, col_idx]), col_idx] = median
        test_data_filled[np.isnan(test_data[:, col_idx]), col_idx] = median
    
    return train_data_filled, test_data_filled

def extract_ids_and_features(data):
    """Extracts IDs and features from the dataset."""
    ids = data[:, 0]  # Extract the Id column
    features = data[:, 1:]  # Extract the features (excluding Id column)
    return ids, features

def create_y_mapping(y_data):
    """Creates a mapping from Ids to y values."""
    y_ids = y_data[:, 0]
    y_values = y_data[:, 1]
    return dict(zip(y_ids, y_values))

def align_y_values(x_ids, y_mapping):
    """Aligns y values to the x_ids using the provided mapping."""
    return np.array([y_mapping[id_] for id_ in x_ids])

def validate_data(features, aligned_y_values):
    """Validates that there are no NaNs and the number of samples matches."""
    assert np.isnan(features).sum() == 0, "There should be no NaNs in the dataset"
    assert features.shape[0] == aligned_y_values.shape[0], (
        "Number of samples in the dataset and labels should be equal but found {} and {}".format(
            features.shape[0], aligned_y_values.shape[0]
        )
    )

def compute_correlation_matrix(features, y_values):
    """Computes the correlation matrix for the features and y values."""
    data_with_y = np.column_stack((features, y_values))
    return np.corrcoef(data_with_y.T)

def filter_features_by_correlation(corr_to_y, feature_headers, threshold):
    """Filters features based on correlation with y values above the specified threshold."""
    columns_to_keep = [i for i, corr_value in enumerate(corr_to_y) if abs(corr_value) >= threshold]
    filtered_features = [feature_headers[i] for i in columns_to_keep]
    return columns_to_keep, filtered_features

def sort_features_by_correlation(filtered_corr, filtered_feature_headers):
    """Sorts features based on the magnitude of their correlation."""
    sorted_indices = np.argsort(np.abs(filtered_corr))[::-1]
    sorted_features = [filtered_feature_headers[i] for i in sorted_indices]
    sorted_corr = filtered_corr[sorted_indices]
    return sorted_features, sorted_corr

def validate_feature_headers(test_headers, train_headers):
    """Validates that the feature headers in test data match those in training data."""
    assert test_headers[1:] == train_headers, "Feature headers in test data do not match training data"

def filter_test_features(test_features, columns_to_keep):
    """Filters test features based on the columns to keep."""
    return test_features[:, columns_to_keep]

def reconstruct_cleaned_data(ids, filtered_features):
    """Reconstructs the cleaned data with Ids and filtered features."""
    return np.column_stack((ids, filtered_features))

def update_headers(filtered_feature_headers):
    """Updates the headers to include 'Id' and filtered feature names."""
    return ['Id'] + filtered_feature_headers