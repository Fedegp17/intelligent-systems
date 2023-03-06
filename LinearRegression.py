import numpy as np

class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr= lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):

            y_pred = np.dot(x, self.weights)+self.bias
            
            dm = (1/n_samples) * np.dot(x.T,(y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.lr *dm
            self.bias = self.bias - self.lr * db
    
    def predict(self, x):

        y_pred = np.dot(x, self.weights)+self.bias
        
        return y_pred
    
    def mse(self, y_test, y_pred):
        
        return np.mean(((y_test-y_pred)**2))
    
    def r_squared(self,y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2