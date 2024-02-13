import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from decision_tree_test import DecisionTree

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTree(max_depth=20)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")