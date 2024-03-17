import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def load_and_process_data(file_path):
    data, meta = arff.loadarff(file_path)
    dataset = np.array(data.tolist(), dtype=np.object)

    # First pass: decode bytes and map categorical values
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            if isinstance(dataset[i, j], bytes):
                dataset[i, j] = dataset[i, j].decode('utf-8')
            if not is_number(dataset[i, j]):
                dataset[i, j] = {'yes': 1, 'no': 0, 
                                 'ckd': 1, 'notckd': 0, 
                                 'normal': 0, 'abnormal': 1, 
                                 'notpresent': 0, 'present': 1, 
                                 'poor': 1, 'good': 0}.get(dataset[i, j].lower(), np.nan)

     # Second pass: replace '?' with NaN, then compute means and replace NaNs
    for j in range(dataset.shape[1] - 1):
        # Replace '?' with np.nan and convert elements to float, if possible
        temp_col = []
        for x in dataset[:, j]:
            if x == '?' or not is_number(x):
                temp_col.append(np.nan)
            else:
                temp_col.append(float(x))
        
        temp_col = np.array(temp_col, dtype=np.float64)  # Convert list to NumPy array

        # Calculate the mean of the column, ignoring NaNs
        mean_value = np.nanmean(temp_col)

        # Replace NaNs in the column with the mean value
        dataset[:, j] = np.where(np.isnan(temp_col), mean_value, temp_col)

    return dataset.astype(np.float64)

def train_and_evaluate(X_train, X_test, y_train, y_test, use_standardization):
    if use_standardization:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    f_measure_train = f1_score(y_train, y_train_pred)

    y_test_pred = model.predict(X_test_scaled)
    f_measure_test = f1_score(y_test, y_test_pred)

    return f_measure_train, f_measure_test

file_path = r"C:\Users\bdefl\Downloads\chronic_kidney_disease_full.arff"

# Load and process the data
dataset = load_and_process_data(file_path)

# Extract features and labels
X = dataset[:, :-1]
y = dataset[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train and evaluate the model for unstandardized data
f_measure_train_unstd, f_measure_test_unstd = train_and_evaluate(X_train, X_test, y_train, y_test, use_standardization=False)

# Train and evaluate the model for standardized data
f_measure_train_std, f_measure_test_std = train_and_evaluate(X_train, X_test, y_train, y_test, use_standardization=True)

# Print the results
print(f"F-measure on Training Set (Unstandardized): {f_measure_train_unstd:.4f}")
print(f"F-measure on Test Set (Unstandardized): {f_measure_test_unstd:.4f}")
print(f"F-measure on Training Set (Standardized): {f_measure_train_std:.4f}")
print(f"F-measure on Test Set (Standardized): {f_measure_test_std:.4f}")

# Plot the results
labels = ['Training Set (Unstandardized)', 'Test Set (Unstandardized)', 'Training Set (Standardized)', 'Test Set (Standardized)']
f_measures = [f_measure_train_unstd, f_measure_test_unstd, f_measure_train_std, f_measure_test_std]

plt.bar(labels, f_measures, color=['blue', 'orange', 'green', 'red'])
plt.title('F-measure Comparison between Standardized and Unstandardized Data')
plt.ylabel('F-measure')
plt.show()
