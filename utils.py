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

class Distances(object):

    def __init__(self, metric):
        self.metric = metric

    def compute(self, x, y):
        metric = self.metric

        if len(x.shape)==1:
            x = x[:, np.newaxis]
        if len(y.shape)==1:
            y = x[:, np.newaxis]

        try:
            method = getattr(self, metric)
            dist = method(x, y)
        except AttributeError:
            raise NotImplementedError('%s distance not implemented.' %self.metric)

        return dist

    @classmethod
    def cosine(cls, x, y):
        def norm(x):
            sum_sq=x.dot(x.T)
            norm=np.sqrt(sum_sq)
            return(norm)

        xy = x.dot(y.T)
        dist = xy/(norm(x)*norm(y))
        return 1-dist

    @classmethod
    def euclidean(cls, x, y):

        dist = np.sqrt(np.sum(np.power(y-x, 2)))

        return dist

    @classmethod
    def mahalanobis(cls, x, y, inv_cov):

        # cov = np.cov(x, y)

        # inv_cov = np.linalg.inv(cov)
        diff = (x-y)

        dist = np.sqrt( diff.T.dot(inv_cov).dot(diff) )

        return dist

    @classmethod
    def chebyshev(cls, x, y):

        dist = np.max(np.absolute(x-y))

        return dist

    @classmethod
    def manhattan(cls, x, y):

        return sum(np.absolute(x-y))
