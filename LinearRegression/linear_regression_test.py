import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=100):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        #init paramaters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # loop through the number of iters
        for _ in range(self.n_iters):
            # compute for y_predict using y = mx + b
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # calculate the integral of dw
            dw = (1/ n_samples) * np.dot(X.T, (y_predicted - y))
            
            #claculate the integral of db
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            #update the weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        
        return y_predicted