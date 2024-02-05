from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# listing the colors
cmap = ListedColormap(["#FF0000", "#00FF00"])

from naive_bayes_test import NaiveBayes


# load the data from the make classification dataset
X, y = make_classification(n_samples=10000, n_features=200, n_classes=2, random_state=123)

# showing the features in a scatter plot
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k', s=20)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb = NaiveBayes()

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print(accuracy_score(y_test, y_pred) * 100)