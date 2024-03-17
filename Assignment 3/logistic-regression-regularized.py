import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
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


def train_and_evaluate(X_train, X_test, y_train, y_test, lambda_values):
    f_measure_train = []
    f_measure_test = []

    for lambda_reg in lambda_values:
        model = LogisticRegression(C=1/np.exp(lambda_reg), solver='liblinear', random_state=42)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        f_measure_train.append(f1_score(y_train, y_train_pred))
        y_test_pred = model.predict(X_test)
        f_measure_test.append(f1_score(y_test, y_test_pred))

    return f_measure_train, f_measure_test

file_path = r"C:\Users\bdefl\Downloads\chronic_kidney_disease_full.arff"

# Load and process the data
dataset = load_and_process_data(file_path)

# Extract features and labels
X = dataset[:, :-1]
y = dataset[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Set up lambda values
lambda_values = np.arange(-2, 4.2, 0.2)

# Train and evaluate the model
f_measure_train, f_measure_test = train_and_evaluate(X_train, X_test, y_train, y_test, lambda_values)

# Plot the results
plt.plot(lambda_values, f_measure_train, label='Training Set')
plt.plot(lambda_values, f_measure_test, label='Test Set')
plt.title('F-measure vs. Regularization Parameter (λ)')
plt.xlabel('Regularization Parameter (log scale)')
plt.ylabel('F-measure')
plt.legend()
plt.show()
