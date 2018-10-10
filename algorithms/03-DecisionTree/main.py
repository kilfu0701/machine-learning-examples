import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Importing data
def importdata():
    balance_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data', sep = ',', header = None)
    print(len(balance_data))
    print(balance_data.shape)
    print(balance_data.head())
    return balance_data

def splitdataset(balance_data):
    x = balance_data.values[: , 1: 5]
    y = balance_data.values[: , 0]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
    return x, y, x_train, x_test, y_train, y_test

# Training with giniIndex
def train_using_gini(x_train, x_test, y_train):
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth = 3, min_samples_leaf = 5)
    clf_gini.fit(x_train, y_train)
    return clf_gini

# Training with entropy
def train_using_entropy(x_train, x_test, y_train):
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)
    clf_entropy.fit(x_train, y_train)
    return clf_entropy

# Making predictions
def prediction(x_test, clf_object):
    y_pred = clf_object.predict(x_test)
    print(f"Predicted values: {y_pred}")
    return y_pred

# Calculating accuracy
def cal_accuracy(y_test, y_pred):
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred) * 100)
    print(classification_report(y_test, y_pred))




data = importdata()

x, y, x_train, x_test, y_train, y_test = splitdataset(data)
clf_gini = train_using_gini(x_train, x_test, y_train)
clf_entropy = train_using_entropy(x_train, x_test, y_train)
y_pred_gini = prediction(x_test, clf_gini)

cal_accuracy(y_test, y_pred_gini)

y_pred_entropy = prediction(x_test, clf_entropy)

cal_accuracy(y_test, y_pred_entropy)
