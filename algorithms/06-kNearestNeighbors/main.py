import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
x = iris.data
y = iris.target
logreg = LogisticRegression()
logreg.fit(x, y)
logreg.predict(x)

y_pred=logreg.predict(x)
print(len(y_pred))

metrics.accuracy_score(y, y_pred)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x, y)
y_pred = knn.predict(x)
metrics.accuracy_score(y, y_pred)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x, y)
y_pred=knn.predict(x)
metrics.accuracy_score(y, y_pred)

print(x.shape)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=4)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = knn.predict(x_test)
metrics.accuracy_score(y_test, y_pred)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
metrics.accuracy_score(y_test, y_pred)

k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

print(scores)


plt.plot(k_range,scores)
plt.xlabel('k for kNN')
plt.ylabel('Testing Accuracy')
plt.show()

