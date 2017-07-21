import pandas as pd                                                  
import numpy as np                                             
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab                             
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
import copy  

# utils

def log_sum_exp(Z):
    ''' Compute log(\sum_i exp(Z_i)) for some array Z.'''
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))

def initialize_clusters(data, k,
    init_covariances=None,
    init_weights=None):
    if isinstance(data, pd.DataFrame):
        # a dataframe is passed, it will be converted to a numpy array
        print('A DataFrame was passed. It will be converted to a NumPy array.')
        data = data.values

    init_means = [data[x] for x in np.random.choice(len(data), k, replace=False)]
    
    # if initial covariances/weights are not provided, set 
    # equal weights and each covariance matrix to be a
    # a diagonal matrix with sample variances on the diagonal
    if init_covariances is None:
        cov = np.diag([data[:, col].var() for col in range(data.shape[1])])
        init_covariances = [cov for i in range(k)]
        init_weights = [1./k for i in range(k)]

    init_dict = {'means': init_means,
                'covs': init_covariances,
                'weights': init_weights}

    return init_dict

### EM ALGORITHM ###


def EM(data, init_means, init_covariances, 
    init_weights, maxiter=1000, thresh=1e-4):
    '''
    data: array-like of array-likes (e.g. lists or numpy arrays)
    init_means: list of means
    init_covariances: list of covariance matrices (one for each cluster)
    init_weights: list of initial mixture weights
    '''
    
    if isinstance(data, pd.DataFrame):
        # a dataframe is passed, it will be converted to a numpy array
        print('A DataFrame was passed. It will be converted to a NumPy array.')
        data = data.values
    
    # Make copies of initial parameters, which we will update during each iteration
    means = init_means[:]
    covariances = init_covariances[:]
    weights = init_weights[:]
    
    num_data = len(data)
    num_dim = len(data[0]) # dimensionality of observations
    num_clusters = len(means)
    
    resp = np.zeros((num_data, num_clusters)) # responsibility matrix
    ll = loglikelihood(data, weights, means, covariances) # initial log-likelihood
    ll_trace = [ll]
    
    for i in range(maxiter):
        if i % 5 == 0:
            print("Iteration %s" % i)
        
######## E-STEP: compute responsibilities
        # Update resp matrix so that resp[j, k] is the responsibility of cluster k for data point j.
        for j in range(num_data):
            for k in range(num_clusters):
                # normalization_term = np.sum([
                #        weights[x]*multivariate_normal.pdf(data[j], mean=means[x], cov=covariances[x]) for x in range(num_clusters)
                #    ])
                resp[j, k] = weights[k]*multivariate_normal.pdf(data[j], mean=means[k], cov=covariances[k])
        row_sums = resp.sum(axis=1)[:, np.newaxis]
        resp = resp / row_sums # normalize over all possible cluster assignments

######## M-STEP
        # Compute the total responsibility assigned to each cluster, which will be useful when 
        # implementing M-steps below. These are called N^{soft}
        counts = np.sum(resp, axis=0)
        
        for k in range(num_clusters):
            
            # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
            weights[k] = counts[k] / num_data
            
            # Update means for cluster k using the M-step update rule for the mean variables.
            # This will assign the variable means[k] to be our estimate for \hat{\mu}_k.
            weighted_sum = 0
            for j in range(num_data):
                weighted_sum += resp[j, k]*data[j] # NB: this will be [num_dim]-dimensional array
            means[k] = weighted_sum / counts[k] # NB: this will be [num_dim]-dimensional array
            
            # Update covariances for cluster k using the M-step update rule for covariance variables.
            # This will assign the variable covariances[k] to be the estimate for \hat{Sigma}_k.
            weighted_sum = np.zeros((num_dim, num_dim))
            for j in range(num_data):
                weighted_sum += resp[j,k]*np.outer(data[j] - means[k], data[j] - means[k])
            covariances[k] = weighted_sum / counts[k]
          
        
        # Compute the loglikelihood at this iteration
        ll_latest = loglikelihood(data, weights, means, covariances)
        ll_trace.append(ll_latest)
        
        # Check for convergence in log-likelihood and store
        # NB: since EM is a coordinate-ascent algorithm, loglikelihood cannot fall upon a new iteration 
        if (ll_latest - ll) < thresh and ll_latest > -np.inf:
            break
        ll = ll_latest
    
    if i % 5 != 0:
        print("Iteration %s" % i)
    
    results_dict = {'weights': weights, 'means': means, 'covs': covariances, 'loglik': ll_trace, 'responsibilities': resp}

    return results_dict

def loglikelihood(data, weights, means, covs):
    ''' Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters. '''
    num_clusters = len(means)
    num_dim = len(data[0])
    
    ll = 0
    for d in data:
        
        Z = np.zeros(num_clusters)
        for k in range(num_clusters):
            
            # Compute (x-mu)^T * Sigma^{-1} * (x-mu)
            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))
            
            # Compute loglikelihood contribution for this data point and this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= 1/2. * (num_dim * np.log(2*np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)
            
        # Increment loglikelihood contribution of this data point across all clusters
        ll += log_sum_exp(Z)
        
    return ll

def assign_clusters(data, results_dict):
    '''
    data: array-like of array-likes (e.g. lists or numpy arrays)
    results_dict: a dictionary of results from the EM algorithm

    The function assigns observations of the provided data
    to their most (probabilistically) likely clusters.

    Returns a DataFrame with cluster assignments, their 
    corresponding probabilities and original observations.
    '''
    if isinstance(data, pd.DataFrame):
        # a dataframe is passed, it will be converted to a numpy array
        print('A DataFrame was passed. It will be converted to a NumPy array.')
        data = data.values

    means = results_dict['means']
    covariances = results_dict['covs']
    N = len(data)
    K = len(means)

    assignments = [0]*N
    probs = [0]*N

    for i in range(N):
        # Compute the score of data point i under each Gaussian component:
        p = np.zeros(K)
        for k in range(K):
            p[k] = multivariate_normal.pdf(data[i], means[k], covariances[k])
            
        # Compute assignments of each data point to a given cluster based on the above scores:
        assignments[i] = np.argmax(p)
        
        # For data point i, store the corresponding score under this cluster assignment:
        probs[i] = max(p)

    clustered_df = pd.DataFrame.from_dict({'assignments': assignments, 
                                            'probs':probs,
                                            'data': data})

    return clustered_df

# Plotting

def plot_contours(data, means, covs, title=''):
    '''
    data: array-like of array-likes (e.g. lists or numpy arrays)
    means: list of means for each mixture distribution
    covs: list of covariance matrices for each mixture distribution
    title: chart title
    '''
    plt.figure()
    plt.plot([x[0] for x in data], [y[1] for y in data],'ko') # data

    delta = 0.025
    k = len(means)
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    ### COORDINATE MATRICES FROM COORDINATE VECTORS
    ### every possible coordinate pair from provided x,y is generated
    X, Y = np.meshgrid(x, y)
    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        sigmax = np.sqrt(cov[0][0])
        sigmay = np.sqrt(cov[1][1])
        sigmaxy = cov[0][1]/(sigmax*sigmay)
        Z = mlab.bivariate_normal(X, Y, sigmax, sigmay, mean[0], mean[1], sigmaxy)
        ### THE PROJECTION OF A 3D FUNCTION ONTO 2D SPACE
        plt.contour(X, Y, Z, colors = col[i])
        plt.title(title)
    plt.rcParams.update({'font.size':16})
    # plt.tight_layout()
    plt.show()

def plot_logl(results_dict):
    loglikelihoods = results_dict['loglik']

    plt.plot(range(len(loglikelihoods)), loglikelihoods, linewidth=4)
    plt.xlabel('Iteration')
    plt.ylabel('Log-likelihood')
    plt.rcParams.update({'font.size':16})
    # plt.tight_layout()
    plt.show()

