import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

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

def train_and_evaluate_svm(X_train, X_test, y_train, y_test, kernel='linear'):
    svm_model = SVC(kernel=kernel)
    svm_model.fit(X_train, y_train)

    y_train_pred = svm_model.predict(X_train)
    f_measure_train = f1_score(y_train, y_train_pred)

    y_test_pred = svm_model.predict(X_test)
    f_measure_test = f1_score(y_test, y_test_pred)

    return f_measure_train, f_measure_test

def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test):
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    y_train_pred = rf_model.predict(X_train)
    f_measure_train = f1_score(y_train, y_train_pred)

    y_test_pred = rf_model.predict(X_test)
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

# Train and evaluate Linear SVM
f_measure_train_linear_svm, f_measure_test_linear_svm = train_and_evaluate_svm(X_train, X_test, y_train, y_test, kernel='linear')

# Train and evaluate RBF SVM
f_measure_train_rbf_svm, f_measure_test_rbf_svm = train_and_evaluate_svm(X_train, X_test, y_train, y_test, kernel='rbf')

# Train and evaluate Random Forest
f_measure_train_rf, f_measure_test_rf = train_and_evaluate_random_forest(X_train, X_test, y_train, y_test)

# Scale values for better visibility
max_value = max(max(f_measure_train_linear_svm, f_measure_test_linear_svm),
                max(f_measure_train_rbf_svm, f_measure_test_rbf_svm),
                max(f_measure_train_rf, f_measure_test_rf))

# Plot the results with scaled y-axis
labels = ['Linear SVM', 'RBF SVM', 'Random Forest']
f_measure_train_values = [f_measure_train_linear_svm, f_measure_train_rbf_svm, f_measure_train_rf]
f_measure_test_values = [f_measure_test_linear_svm, f_measure_test_rbf_svm, f_measure_test_rf]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, f_measure_train_values, width, label='Training Set', color='b')
rects2 = ax.bar(x + width/2, f_measure_test_values, width, label='Test Set', color='g')

# Set the same scale for both training and test F-measures
ax.set_ylim([0, max_value + 0.1])

# Add labels, title, and legend
ax.set_ylabel('F-measure')
ax.set_title('F-measure by Algorithm and Dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Show the plot
plt.show()