# testing our linear regression with the make_regression dataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from linear_regression_test import LinearRegression

# get our dataset 
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

# split our data to training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


# call our model
model = LinearRegression(n_iters=10000, lr=0.1)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(f"The mean squared error: {mean_squared_error(y_test, pred)}")

# plot the regression grah
plt.figure()
plt.scatter(X_test, y_test, color="black")
plt.plot(X_test, pred, color="blue", linewidth=3)
plt.show()

# show a plot of the data
# plt.figure()
# plt.scatter(X[:, 0], y, s=30, marker="o", edgecolors="b")
# plt.show()