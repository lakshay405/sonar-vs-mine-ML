import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset into a Pandas DataFrame
sonar_data = pd.read_csv('data.csv', header=None)

# Displaying the first few rows of the dataframe
sonar_data.head()

# Checking the shape of the dataset (number of rows and columns)
sonar_data.shape

# Displaying statistical summary of the dataset
sonar_data.describe()

# Counting the occurrences of each class in the target column
sonar_data[60].value_counts()

# Calculating mean values grouped by the target column
sonar_data.groupby(60).mean()

# Separating the data into features (X) and target labels (y)
X = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]

# Displaying the features and target labels
print("Features (X):\n", X)
print("Target (y):\n", y)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)

# Displaying the shapes of training and test sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Model building and training with Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Model building and training with Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Accuracy evaluation on training data using Logistic Regression
y_train_predictions = logistic_model.predict(X_train)
training_accuracy = accuracy_score(y_train, y_train_predictions)
print('Accuracy on training data:', training_accuracy)

# Building a predictive system for new input data
input_data = np.array([0.02,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.066,0.2273,0.31,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.555,0.6711,0.6415,0.7104,0.808,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.051,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.018,0.0084,0.009,0.0032])
input_data_reshaped = input_data.reshape(1, -1)

prediction = logistic_model.predict(input_data_reshaped)

print("Prediction:", prediction[0])

if prediction[0] == 'R':
    print('The object is a Rock')
else:
    print('The object is a Mine')
