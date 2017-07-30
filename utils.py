import pandas as pd
import numpy as np

class Preprocess(object):

    def __init__(self):
        self.norms = None
        self.weights = None

    def normalize(self, X_train):
        norms = np.linalg.norm(X_train, axis=0)
        X_normalized = X_train / norms

        self.norms = norms

        return X_normalized

    def add_constant(self, X_train):
        if isinstance(X_train, pd.DataFrame):
            X_train['intercept'] = 1

        else:
            X_train = np.append(np.ones(X_train.shape[0])[:, np.newaxis],
                                X_train, axis=1)

        return X_train
