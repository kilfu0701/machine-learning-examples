import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Test set; straight line with Gaussian noise
xmin, xmax = -7, 7
n_samples = 77
np.random.seed(0)
x = np.random.normal(size=n_samples)
y = (x > 0).astype(np.float)
x[x > 0] *= 3
x += .4 * np.random.normal(size=n_samples)
x = x[:, np.newaxis]

# Classifier
clf = linear_model.LogisticRegression(C=1e4)
clf.fit(x,y)

plt.figure(1, figsize=(3, 4))
plt.clf()
plt.scatter(x.ravel(), y, color='lavender', zorder=17)

x_test = np.linspace(-7, 7, 277)
def model(x):
     return 1 / (1 + np.exp(-x))

loss = model(x_test * clf.coef_+clf.intercept_).ravel()
plt.plot(x_test, loss, color='pink', linewidth=2.5)

ols = linear_model.LinearRegression()
ols.fit(x,y)

plt.plot(x_test, ols.coef_ * x_test + ols.intercept_, linewidth=1)
plt.axhline(.4, color='.4')
plt.ylabel('y')
plt.xlabel('x')

plt.xticks(range(-7, 7))
plt.yticks([0, 0.4, 1])
plt.ylim(-.25, 1.25)
plt.xlim(-4, 10)
plt.legend(('Logistic Regression', 'Linear Regression'), loc='lower right', fontsize='small')
plt.show()
