'''
Logistic Regression with L2 penalty via Stochastic Gradient Descent.

email: dat.nguyen at cantab.net
'''

import numpy as np
from math import sqrt

class LogisticRegSGD(object):
	def __init__(self, step_size=1e-4, batch_size=1, maxiter=100,
		regularize=True, l2_penalty=1):
		self.step_size = step_size
		self.batch_size = batch_size
		self.maxiter = maxiter
		self.regularize = regularize
		self.l2_penalty = l2_penalty

	def _compute_avgll(self, X, y, weights):
    
	    indicator = (y==1)
	    scores = np.dot(X, weights)
	    logexp = np.log(1. + np.exp(-scores))
	    
	    # Simple check to prevent overflow
	    mask = np.isinf(logexp)
	    logexp[mask] = -scores[mask]
	    
	    avg_ll = np.sum((indicator-1)*scores - logexp)/len(X)
	    
	    return avg_ll

	def _compute_gradient(self, errors, feature, weights): 
    
	    gradient = np.dot(feature, errors)

	    return gradient

	def predict_proba(self, X, weights):
		score = np.dot(X, weights)
    
	    predictions = 1. / (1.+np.exp(-score))    
	    return predictions

	def fit(self, X, y, initial_weights, 
	seed=None):
		step_size = self.step_size
		batch_size = self.batch_size
		maxiter = self.maxiter

	    log_likelihood_all = []
	    weights = np.array(initial_weights)
	    if seed is not None:
	    	np.random.seed(seed=seed)
	    
	    # shuffle data
	    permutation = np.random.permutation(len(X))
	    X = X[permutation,:]
	    y = y[permutation]
	    
	    i = 0 
	    for itr in range(maxiter):

	        predictions = predict_proba(X[i:i+batch_size, :], weights)
	        
	        indicator = y[i:i+batch_size]==+1
	        
	        errors = indicator - predictions

	        for j in range(len(weights)): # loop over each coefficient
	            
	            derivative = self._compute_gradient(errors, X[i:i+batch_size, j], weights)
	            if self.regularize and j>0:
					derivative -= 2*self.l2_penalty*weights[j]
	            
	            # the **normalization constant** (1./batch_size)
	            weights[j] += step_size * derivative * 1./batch_size
	        
	        # Checking whether log likelihood is increasing
	        # Print the log likelihood over the *current batch*
	        lp = self._compute_avgll(X[i:i+batch_size,:], y[i:i+batch_size],
	                                        weights)
	        log_likelihood_all.append(lp)
	        if itr <= 15 or (itr <= 1000 and itr % 100 == 0) or (itr <= 10000 and itr % 1000 == 0) \
	         or itr % 10000 == 0 or itr == maxiter-1:
	            data_size = len(X)
	            print('Iteration %*d: Average log likelihood (of data points in batch [%0*d:%0*d]) = %.8f' % \
	                (int(np.ceil(np.log10(maxiter))), itr, \
	                 int(np.ceil(np.log10(data_size))), i, \
	                 int(np.ceil(np.log10(data_size))), i+batch_size, lp))
	        
	        # if we made a complete pass over data, shuffle and restart
	        i += batch_size
	        if i+batch_size > len(X):
	            permutation = np.random.permutation(len(X))
	            X = X[permutation,:]
	            y = y[permutation]
	            i = 0

	    self.log_likelihood_all = log_likelihood_all
	    self.weights = weights
	    
	    # return weights, log_likelihood_all

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
