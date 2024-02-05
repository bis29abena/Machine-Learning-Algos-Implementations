from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from logistic_regression_test import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd

# load the dataset from the sklearn
bc = datasets.load_breast_cancer()

# get your feature data and predicted labels
X, y = bc.data, bc.target

# split the data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# train your data with the model
model = LogisticRegression()

model.fit(X, y)

predictions = model.predict(X_test)

print(f"Accuaracy Score: {accuracy_score(y_test, predictions)}\n")

print("Classification Report\n")
print(classification_report(y_test, predictions, target_names=bc.target_names))



