"""
Naive Bayes - 單純貝氏分類器

Naive Bayes is a classification method which is based on Bayes’ theorem.
This assumes independence between predictors. A Naive Bayes classifier will assume that a feature in a class is unrelated to any other.
Consider a fruit. This is an apple if it is round, red, and 2.5 inches in diameter.
A Naive Bayes classifier will say these characteristics independently contribute to the probability of the fruit being an apple.
This is even if features depend on each other.

Methods:
- GaussianNB
    It is used in classification and it assumes that features follow a normal distribution.

- MultinomialNB
    It is used for discrete counts.
    For example, let’s say, we have a text classification problem. Here we can consider bernoulli trials which is one step further and
    instead of “word occurring in the document”, we have “count how often word occurs in the document”, you can think of it as
    “number of times outcome number x_i is observed over the n trials”.

- BernoulliNB
    The binomial model is useful if your feature vectors are binary (i.e. zeros and ones).
    One application would be text classification with ‘bag of words’ model where the 1s & 0s are
    “word occurs in the document” and “word does not occur in the document” respectively.

Refs:
http://cpmarkchang.logdown.com/posts/193470-natural-language-processing-naive-bayes-classifier
https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
gnb = GaussianNB()
mnb = MultinomialNB()
y_pred_gnb = gnb.fit(x_train, y_train).predict(x_test)
cnf_matrix_gnb = confusion_matrix(y_test, y_pred_gnb)
print(cnf_matrix_gnb)

y_pred_mnb = mnb.fit(x_train, y_train).predict(x_test)
cnf_matrix_mnb = confusion_matrix(y_test, y_pred_mnb)
print(cnf_matrix_mnb)
