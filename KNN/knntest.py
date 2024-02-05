import numpy as np
from collections import Counter

def euclidea_distance(x1, x2):
    """
        Calculate the euclidean distance 
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    # implementing the fit for the training data and training labels
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    # implenting the predict method for predictions
    def predict(self, X):
        predicted_labels = [self.__pred(x) for x in X]
        return predicted_labels
        
    def __pred(self, x):
        # compute the distance
        distance = [euclidea_distance(x, x_train) for x_train in self.X_train]
        
        # get the k nearest sample, labels
        k_indices = np.argsort(distance)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # majority vote, most common class labels
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]