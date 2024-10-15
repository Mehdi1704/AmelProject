import numpy as np
import csv

# Columns to drop
variables2 = ["Name", "Email"]

file_path = 'dataset/random_data.csv'

# Read the data using the csv module
with open(file_path, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# Extract header and data separately
header = data[0]
data_rows = data[1:]

# Get indices of columns to drop
indices_to_drop = [header.index(col) for col in variables2 if col in header]

# Remove columns from header
header = [col for idx, col in enumerate(header) if idx not in indices_to_drop]

# Remove columns from data rows
processed_data = []
for row in data_rows:
    new_row = [value for idx, value in enumerate(row) if idx not in indices_to_drop]
    processed_data.append(new_row)

# Convert processed_data to a numpy array with object data type
x_train = np.array(processed_data, dtype=object)

# Print the first 5 rows after deletion for validation
print(x_train[:5])
print(header)
