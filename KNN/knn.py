# testing our data with the iris dataset from scikit learn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from knntest import KNN
from matplotlib.colors import ListedColormap

# listing the colors
cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

# load the iris dataset from the model
iris = datasets.load_iris()

# assign your training data and labels
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = KNN()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

score = round(accuracy_score(y_test, pred) * 100, 2)

print(f"accuracy_score: {score}\n" )
print("Classification Report")
print(classification_report(y_test, pred))


# showing the features in a scatter plot
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k', s=20)
# plt.show()