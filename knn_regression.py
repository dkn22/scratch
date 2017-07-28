'''
K-nearest neighbours regression.

Currently, the algorithm uses linear search for
nearest neighbour search, which is not scalable/efficient 
for very large datasets.

TBC: approximate search (locality sensitive hashing) 
and KD tree implementation.

email: dat.nguyen at cantab.net
'''
import bisect
import numpy as np

class KnnRegression(object):
    '''
    KNN regression class.

    The model finds k nearest neighbours for
    test data and outputs the prediction as the
    mean of k neighbours' target values.
    '''

    def __init__(self, num_neighbors, normalize_features=False):
        self.num_neighbors = num_neighbors
        self.normalize_features = normalize_features
        self.norms = None

    def choose_k(self, X_train, y_train, X_valid, y_valid,
                 k_list=np.arange(1,10)):
        '''
        X_train: dataframe/SFrame of features
        y_train: array-like of target values
        X_valid: dataframe/SFrame/matrix of observations
                to be used for cross-validation.
        y_valid: array-like of validation target values

        Stores the optimal K that minimises squared error
        (residual sum of squares).
        '''

        assert all(k_list > 0)

        rss_valid = []
        for k in k_list:
            self.num_neighbors = k
            predictions = self.predict(X_train, y_train, X_valid)
            rss_validation = np.sum((y_valid - predictions)**2)
            rss_valid.append(rss_validation)

        self.num_neighbors = np.argsort(rss_valid)[0]
        print('Optimal K: ', np.argsort(rss_valid)[0])


    def k_nearest_neighbors(self, X_train, x_query):
        '''
        X_train: dataframe/SFrame of features
        x_query: feature vector
        '''

        k = self.num_neighbors

        # if self.normalize_features:
        #   X_train = self._normalize_features(X_train)

        # initialization: a sorted array of first k obs
        neighbors = np.sort(X_train[:k]) 
        neighbors_idx = list(range(k))

        distances = self._compute_distances(X_train, x_query)
        neighbors_distances = np.sort(distances[:k])
        n = X_train.shape[0]

        for i in range(k+1, n):
            if distances[i] < neighbors_distances[-1]:

                # index where to insert the new neighbour
                j = bisect.bisect(neighbors_distances, distances[i])
                neighbors_distances[j+1:k] = neighbors_distances[j:k-1]
                neighbors[j+1:k] = neighbors[j:k-1]
                neighbors_idx[j+1:k] = neighbors_idx[j:k-1]
                
                # insert new neighbor
                neighbors[j] = X_train[i]
                neighbors_distances[j] = distances[i]
                neighbors_idx[j] = i

        return neighbors, neighbors_idx

    def _compute_distances(self, X_train, x_query):
        '''
        X_train: dataframe/SFrame of features
        x_query: feature vector
        '''
        diff = X_train[:] - x_query
        distances = np.sqrt(np.sum(diff**2, axis=1))

        return distances

    def _predict_query(self, X_train, y_train, x_query):
        '''
        X_train: dataframe/SFrame of features
        y_train: array-like of target values
        x_query: observation to predict
        '''
        
        # k = self.num_neighbors

        neighbors, neighbors_idx = self.k_nearest_neighbors(X_train, x_query)
        neighbors_output = y_train[neighbors_idx]
        prediction = neighbors_output.mean()

        return prediction

    def predict(self, X_train, y_train, X_test):
        '''
        X_train: dataframe/SFrame of features
        y_train: array-like of target values
        X_test: dataframe/SFrame/matrix of observations to predict
        '''
        predictions = []
        if self.normalize_features:
            X_train = self._normalize_features(X_train)
            X_test = X_test / self.norms

        for i in range(X_test.shape[0]):
            prediction = self._predict_query(X_train, y_train, X_test[i])
            predictions.append(prediction)

        return predictions

    def _normalize_features(self, X_train):
        norms = np.linalg.norm(X_train, axis=0)
        X_normalized = X_train / norms

        self.norms = norms

        return X_normalized 
