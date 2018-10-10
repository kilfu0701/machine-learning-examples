"""
Support Vector Machines - 支援向量機

SVM is a supervised classification Python Machine Learning algorithms that plots a line that divides different categories of your data.
In this ML algorithm, we calculate the vector to optimize the line.
This is to ensure that the closest point in each group lies farthest from each other.
While you will almost always find this to be a linear vector, it can be other than that.

Wiki:
https://zh.wikipedia.org/wiki/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA

SVM優點：
- 切出來的線很漂亮，擁有最大margin的特性
- 可以很容易透過更換Kernel，做出非線性的線（非線性的決策邊界）

SVM缺點：
- 效能較不佳，由於時間複雜度為O(n²)當有超過一萬筆資料時，運算速度會慢上許多

Refs:
https://medium.com/@yehjames/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-4%E8%AC%9B-%E6%94%AF%E6%8F%B4%E5%90%91%E9%87%8F%E6%A9%9F-support-vector-machine-%E4%BB%8B%E7%B4%B9-9c6c6925856b
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

x, y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.40)

plt.scatter(x[:,0], x[:,1], c=y, s=50, cmap='plasma')
plt.show()


xfit = np.linspace(-1, 3.5)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='plasma')

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AFFEDC', alpha=0.4)

plt.xlim(-1, 3.5)
plt.show()
