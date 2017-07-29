'''
LASSO Regression via Coordinate Descent Algorithm.

email: dat.nguyen at cantab.net
'''
import numpy as np
import pandas as pd

class LassoRegression(object):
    '''
    LASSO regression, also known as L1-regularized regression.
    '''

    def __init__(self, l1_penalty=1, normalize_features=True):
        self.normalize_features = normalize_features
        self.norms = None
        self.weights = None
        self.l1_penalty = l1_penalty

    def _normalize_features(self, X_train):
        norms = np.linalg.norm(X_train, axis=0)
        X_normalized = X_train / norms

        self.norms = norms

        return X_normalized

    def _update_weight(self, weight_idx, X_train, y_train, weights):
        '''
        weight_idx: index of the feature column;
        X_train: feature matrix;
        y_train: vector of target values;
        weights: feature weights.

        Update a weight using subgradients.
        '''
        i = weight_idx
        l1_penalty = self.l1_penalty

        prediction = X_train.dot(weights)
        ro_i = np.sum(X_train[:, i]*(y_train - prediction + weights[i]*X_train[:, i]))
        
        # do not regularize the intercept
        if i == 0:
            new_weight_i = ro_i

        elif ro_i < -l1_penalty/2.:
            new_weight_i = ro_i + l1_penalty/2
        elif ro_i > l1_penalty/2.:
            new_weight_i = ro_i - l1_penalty/2
        else:
            new_weight_i = 0.
        
        return new_weight_i

    def fit(self, X_train, y_train, 
            initial_weights=None,
            add_intercept=True,
            tolerance=1.0,
            maxiter=10000):
        '''
        X_train: feature matrix;
        y_train: vector of target values;
        initial_weights: initial feature weights;
        add_intercept: whether to add a column of ones to X;
        tolerance: convergence criterion;
        maxiter: maximum number of coordinate descent iterations.

        Train a LASSO regression via coordinate descent.
        '''

        if self.normalize_features:
            X_train = self._normalize_features(X_train)

        if initial_weights is None:
            initial_weights = np.zeros(X_train.shape[1])

        if add_intercept:
            if isinstance(X_train, pd.DataFrame):
                X_train['intercept'] = 1

            else:
                X_train = np.append(np.ones(X_train.shape[0])[:, np.newaxis],
                                    X_train, axis=1)

        num_col = X_train.shape[1]
        weights = np.array(initial_weights[:])
        max_step = np.inf

        itr = 0
        while max_step >= tolerance and itr <= maxiter:
            old_weights = weights.copy()

            for i in range(num_col):
                weights[i] = self._update_weight(i, X_train, y_train, weights)

            max_step = max(weights - old_weights)

            itr += 1

        self.weights = weights
        self.selected_features = list(X_train.columns)[weights != 0]

        # return weights

    def predict(self, X_test):
        '''
        X_test: test data.

        Make a prediction by taking a dot product.
        '''

        if self.normalize_features:
            try:
                X_test = X_test / self.norms
            except Exception:
                raise ValueError('Training features need to be normalized first or as part of fitting a training model.')

        predictions = X_test.dot(self.weights)

        return predictions
