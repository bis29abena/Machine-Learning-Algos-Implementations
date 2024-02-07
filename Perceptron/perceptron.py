import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from perceptron_test import Perceptron

X, y = make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model_perceptron = Perceptron()

model_perceptron.fit(X_train, y_train)

pred = model_perceptron.predict(X_test)

# print(pred)

print(f"Perceptron Classification Accuracy = {accuracy_score(y_true=y_test, y_pred=pred)} \n")

print(f"Confusion matrix = {confusion_matrix(y_true=y_test, y_pred=pred)}")


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train)

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = (-model_perceptron.weights[0] * x0_1 - model_perceptron.bias) / model_perceptron.weights[1]
x1_2 = (-model_perceptron.weights[0] * x0_2 - model_perceptron.bias) / model_perceptron.weights[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])

ax.set_ylim([ymin - 3, ymax + 3])

plt.show()