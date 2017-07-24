'''
Logistic regression with L2 penalty and cross-validation.

email: dat.nguyen at cantab.net
'''

import numpy as np
import json
import re
import string
import operator
from math import sqrt


class LogisticRegression(object):
	def __init__(self, regularize=True, step_size=1e-6, maxiter=1000, l2_penalty=1):
		self.regularize = regularize
		self.l2_penalty = l2_penalty
		self.step_size = step_size
		self.maxiter = maxiter
		self.weights = None

	def _sigmoid(self, X, weights):
		score = X.dot(weights)
		proba = 1 / (1 + np.exp(-score))

		return proba

	def _compute_logl(self, X, y, weights):
		indicator = (y == 1)
		score = X.dot(weights)
		if self.regularize:
			logl = np.sum( (indicator - 1)*score - np.log(1 + np.exp(-score)) - self.l2_penalty*np.sum(weights[1:]**2) )
		else:
			logl = np.sum( (indicator - 1)*score - np.log(1 + np.exp(-score)) )

		return logl

	def fit(self, X, y, initial_weights=None, cv=False):

		if initial_weights is None:
			num_param = X.shape[1]
			initial_weights = np.random.normal(0, 0.1, size=(num_param, 1))

		weights = np.asarray(initial_weights)
		indicator = (target == 1)

		for itr in range(maxiter):
			pred_proba = self._sigmoid(X, weights)
			errors = indicator - pred_proba

			gradient = np.dot(errors, X)
			if self.regularize:
				gradient[1:] -= 2*self.l2_penalty*weights[1:]

				weights = weights[0] + step_size*gradient
			else:
				weights = weights + step_size*gradient
			if itr <= 1 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        	or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:

        		logl = self._compute_logl(X, y, weights)
        		print('iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(maxiter))), itr, logl))

        self.weights = weights

        if cv:
        	return weights

    def predict(self, X_test, weights=None):
    	'''
		Classifies an observation as "positive" if
		P(positive class) > 0.5; otherwise "negative".
    	'''
    	if weights is None:
    		weights = self.weights

    	score = X_test.dot(weights)

    	predictions = (score > 0)*1

    	return predictions

   	def fit_CV(self, X, y, X_valid, y_valid,
   		l2_penalty_list=[0, 4, 10, 1e2, 1e3, 1e5],
   		verbose=True):
   		'''
		Fit a ridge regression using cross-validation to find 
		the optimal L2 penalty.
   		'''
   		assert self.regularize == True

   		results = {}
   		for pen in l2_penalty_list:
   			label = 'L2_' + str(pen)
   			weights = self.fit(X, y, cv=True)

   			results[label] = weights

   		metrics = {}
   		for pen, reg in results.items():
   			label = 'L2_' + str(pen)
   			predictions = self.predict(X_valid, weights=results[label])
   			accuracy = sum((predictions == y_valid)*1)/len(predictions)
   			metrics[label] = accuracy

   		sorted_accuracy = sorted(metrics.items(), key=operator.itemgetter(1), reverse=True)
   		best_accuracy = sorted_accuracy[0][1]
   		best_model = sorted_accuracy[0][0]

   		self.weights = results[best_model]

   		self.best_model = best_model

   		if verbose:
   			print('The best model:', self.best_model)