'''
email: dat.nguyen at cantab.net
'''

import numpy as np
from utils import *

class LeastSquaresRegression(Preprocess):
    '''
    OLS Regression. Weights are computed exactly,
    without an optimization procedure.
    '''

    def __init__(self, normalize_features=True):
        Preprocess.__init__(self)
        self.normalize_features = normalize_features
        self.rss = None
        self.ess = None
        self.r_squared = None

    def fit(self, X_train, y_train,
            add_intercept=True):

        if add_intercept:
            X_train = self.add_constant(X_train)

        if self.normalize_features:
            X_train = self.normalize(X_train)

        X_train = np.asarray(X_train)

        try:
            pseudo_inverse = np.linalg.inv(X_train.T.dot(X_train))
        except Exception:
            raise ValueError('Pseudo-inverse could not be computed. Check for collinear features.')

        weights = pseudo_inverse.dot(X_train.T).dot(y_train)

        self.weights = weights

        # residual sum of squares
        self.rss = sum((y_train - self.predict(X_train))**2)
        # explainde sum of squares
        self.ess = sum((self.predict(X_train) - y_train.mean())**2)
        self.r_squared = 1 - self.rss/self.ess

        print('Training completed.')
        print('RSS: ', self.rss)
        print('R_Squared: ', self.r_squared)

    def predict(self, X_test):

        if self.normalize_features:
            X_test = X_test / self.norms

        predictions = X_test.dot(self.weights)

        return predictions
