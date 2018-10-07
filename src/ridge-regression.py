class RidgeRegression(object):
    
    def __init__(self, lamda=10):
        self.lamda = lamda

    def fit(self, X, y):
        # Applying weight vector formula for Ridge regression.
        self.w = np.linalg.inv(X.T.dot(X) + self.lamda*np.eye(X.shape[1])).dot(X.T.dot(y))
        return self.w
    
    def predict(self, X):
        # Return Xw
        return X.dot(self.w)
    
    # Function to compute Root Mean Squared Error.
    def rmse(self, X, y, w):
        # Deviation
        d = y - X.dot(w)
        # Square
        s = d.T.dot(d)
        # Root mean
        error = (s/len(X))**(1/2.0)
        return error
    
    # To calculate Maximum Absolute Deviation
    def mad(self, X, y, w):
        d = y - X.dot(w)
        d = np.absolute(d)
        error = np.sum(d)/len(X)
        return error