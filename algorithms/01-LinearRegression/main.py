import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:,np.newaxis,2]

# splitting data into training and test sets
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

# splitting targets into training and test sets
diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

# Linear regression object
regr = linear_model.LinearRegression()
# Use training sets to train the model
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions
diabetes_y_pred=regr.predict(diabetes_X_test)
regr.coef_

mean_squared_error(diabetes_y_test,diabetes_y_pred)

# Variance score
r2_score(diabetes_y_test,diabetes_y_pred)

plt.scatter(diabetes_X_test, diabetes_y_test, color='lavender')
plt.plot(diabetes_X_test, diabetes_y_pred, color='pink', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
