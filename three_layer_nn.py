'''
Full-batch learning for a three-layer perceptron,
i.e. a three-layer fully-connected feedforward neural net,
for binary classification.

TBC: replace sigmoid activation with softmax and provide
    functionality for arbitrary multi-class classification.

email: dat.nguyen at cantab.net
'''

import numpy as np

class ThreeLayerPerceptron(object):
    def __init__(self, learning_rate=0.5, maxepochs=1e4, 
        convergence_thres=1e-5, hidden_layer=4):
        '''
        learning_rate: parameter regulating size of the update step 
                        in the direction of the gradient.
        maxepochs: number of epochs for which training will run.
        convergence_thres: threshold to assess convergence of parameters.
        hidden_layer: number of hidden units in the hidden layer.
        '''
        self.learning_rate = learning_rate
        self.maxepochs = int(maxepochs)
        self.convergence_thres = 1e-5
        self.hidden_layer = int(hidden_layer)

    def _sigmoid_activation(self, X, theta):
        X = np.asarray(X)
        theta = np.asarray(theta)
        
        # if theta is a one-dimensional vector
        if len(theta.shape) == 1:
            theta = theta[:, np.newaxis]

        try:
            assert theta.T.shape[1] == X.shape[0]
            return 1 / (1 + np.exp(-np.dot(theta.T, X)))
        except AssertionError:
            assert theta.shape[0] == X.shape[1]
            return 1 / (1 + np.exp(-np.dot(X, theta)))

    def _compute_cost(self, X, y):
        l1, l2 = self._feedforward(X) 
        # log loss
        loss = y * np.log(l2) + (1-y) * np.log(1-l2)
        # negative of average error
        return -np.mean(loss)
    
    def _feedforward(self, X):
        '''
        Forward propagation.

        Returns the values of the hidden units (l1)
        and the values of the output unit(s) (l2).
        '''

        # feedforward to the first layer
        l1 = self._sigmoid_activation(X.T, self.theta0).T
        # add a column of ones for bias term
        l1 = np.column_stack([np.ones(l1.shape[0]), l1])
        # activation units are then inputted to the output layer
        l2 = self._sigmoid_activation(l1.T, self.theta1)
        return l1, l2
    
    def fit(self, X, y, verbose=False):
        nobs, ncols = X.shape
        self.theta0 = np.random.normal(0,0.01,size=(ncols,self.hidden_layer))
        self.theta1 = np.random.normal(0,0.01,size=(self.hidden_layer+1,1))
        
        self.costs = []
        cost = self._compute_cost(X, y)
        self.costs.append(cost)
        costprev = cost + self.convergence_thres+1
        counter = 0

        for counter in range(self.maxepochs):
            # feedforward through network
            l1, l2 = self._feedforward(X)

            # Backpropagation

            # gradients wrt weights going to the output layer
            l2_delta = (y-l2) * l2 * (1-l2)

            # gradients wrt weights going to the hidden layer
            l1_delta = l2_delta.T.dot(self.theta1.T) * l1 * (1-l1)

            # Update parameters by averaging gradients and multiplying by the learning rate
            self.theta1 += l1.T.dot(l2_delta.T) / nobs * self.learning_rate
            self.theta0 += X.T.dot(l1_delta)[:,1:] / nobs * self.learning_rate
            
            # Store costs and check for convergence
            counter += 1
            costprev = cost
            cost = self._compute_cost(X, y)
            if verbose and counter % 100 == 0:
                print('Iteration %d, cost = %.2f' %(counter, cost))
            self.costs.append(cost)
            if np.abs(costprev-cost) < self.convergence_thres and counter > 500:
                break

        self.weights = {'hidden_layer': self.theta0,
                        'output_layer': self.theta1}

    def predict(self, X):
        _, y = self._feedforward(X)
        return y
