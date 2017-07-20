'''
An implementation of K-Means with smart initialization.

The script requires the data to be inputted as a sparse SciPy matrix.
This makes this implementation particularly suitable for clustering text documents, 
where the bag-of-words representation is most often a highly sparse matrix.

email: dat.nguyen at cantab.net
'''

import matplotlib.pyplot as plt                               
import numpy as np                            
from scipy.sparse import csr_matrix
from scipy import sparse                
from sklearn.preprocessing import normalize 
from sklearn.feature_extraction.text import CountVectorizer                
from sklearn.metrics import pairwise_distances
import sys      
import os
import time

# utils

def to_sparse_matrix(df):
	'''
	Converts a pandas DataFrame to a SciPy
	sparse matrix.
	'''
	return csr_matrix(df.values)

def text_to_sparse_matrix(raw_docs):
	'''
	Returns a sparse bag-of-words representation
	of text documents.
	'''
	data = CountVectorizer.fit(raw_docs).transform(raw_docs)
	return data

# K-Means algorithm

def kmeans_multiple_runs(data, k, maxiter, num_runs, 
	seed_list=None, verbose=False):
	   '''
    This function runs k-means on given data multiple times 
    and returns the best clustering, as
    measured by heterogeneity.
       
    data: a sparse scipy matrix
    maxiter: maximum number of iterations to run.

    '''
	# assert sparse.isspmatrix_csr(data)
    heterogeneity = {}
    
    min_heterogeneity_achieved = float('inf')
    best_seed = None
    final_centroids = None
    final_cluster_assignment = None
    
    for i in range(num_runs):
        
        # Use UTC time if no seeds are provided 
        if seed_list is not None: 
            seed = seed_list[i]
            np.random.seed(seed)
        else: 
            seed = int(time.time())
            np.random.seed(seed)
        
        # k-means++ initialization
        initial_centroids = smart_initialize(data, k, seed)
        
        # k-means after 'smart' initialization
        centroids, cluster_assignment = kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False)
        
        # To save time, compute heterogeneity only once in the end
        heterogeneity[seed] = compute_heterogeneity(data, k, centroids, cluster_assignment)
        
        if verbose:
            print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
            sys.stdout.flush()
        
        # if current measurement of heterogeneity is lower than previously seen,
        # update the minimum record of heterogeneity.
        if heterogeneity[seed] < min_heterogeneity_achieved:
            min_heterogeneity_achieved = heterogeneity[seed]
            best_seed = seed
            final_centroids = centroids
            final_cluster_assignment = cluster_assignment
    
    # Return the centroids and cluster assignments that minimize heterogeneity.
    return final_centroids, final_cluster_assignment


def kmeans(data, k, initial_centroids, maxiter, 
	record_heterogeneity=None, verbose=False):
    '''
    This function runs k-means on given data and initial set of centroids.
       
	data: sparse scipy matrix.
    maxiter: maximum number of iterations to run.
    record_heterogeneity: (optional) a list, to store the history of heterogeneity as function of iterations
                             if None, do not store the history.
    verbose: if True, print how many data points changed their cluster labels in each iteration
    '''
    # assert sparse.isspmatrix_csr(data)

    centroids = initial_centroids[:]
    prev_cluster_assignment = None
    
    for itr in range(maxiter):        
        if verbose:
            print(itr)
        
        # E-step: cluster assignments using nearest centroids
        cluster_assignment = assign_clusters(data, centroids)
            
        # M-step: a new centroid for each of the k clusters, 
        # averaging all data points assigned to that cluster.
        centroids = revise_centroids(data, k, cluster_assignment)
            
        # if none of the assignments changed, stop training
        if prev_cluster_assignment is not None and \
          (prev_cluster_assignment==cluster_assignment).all():
            break
        
        # Assess how assignments change during training
        if prev_cluster_assignment is not None:
            num_changed = np.sum(prev_cluster_assignment!=cluster_assignment)
            if verbose:
                print('    {0:5d} elements changed their cluster assignment.'.format(num_changed))   
        
        # heterogeneity convergence metric
        if record_heterogeneity is not None:
            score = compute_heterogeneity(data, k, centroids, cluster_assignment)
            record_heterogeneity.append(score)
        
        prev_cluster_assignment = cluster_assignment[:]
        
    return centroids, cluster_assignment

def smart_initialize(data, k, seed=None):
    '''
	data: a csr_sparse matrix
	k: number of clusters

    Use k-means++ to initialize a good set of centroids
    '''

    if seed is not None: # useful for obtaining consistent results
        np.random.seed(seed)
    centroids = np.zeros((k, data.shape[1]))
    
    # Randomly choose the first centroid.
    # Since we have no prior knowledge, choose uniformly at random
    idx = np.random.randint(data.shape[0])
    centroids[0] = data[idx,:].toarray()
    # distances from the first centroid chosen to all the other data points
    distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()
    
    for i in range(1, k):
        # Choose the next centroid randomly, so that the probability for each data point to be chosen
        # is directly proportional to its squared distance from the nearest centroid.
        # Roughtly speaking, a new centroid should be as far as from ohter centroids as possible.
        idx = np.random.choice(data.shape[0], 1, p=distances/sum(distances))
        centroids[i] = data[idx,:].toarray()

        # distances from the centroids to all data points
        distances = np.min(pairwise_distances(data, centroids[0:i+1], metric='euclidean'),axis=1)
    
    return centroids

def get_initial_centroids(data, k, seed=None):
    '''
	data: a csr_sparse matrix
	k: number of clusters

    Randomly chooses k data points as initial centroids.

    '''

    if seed is not None: # useful for obtaining consistent results
        np.random.seed(seed)
    n = data.shape[0] # number of data points
        
    # Pick K indices from range [0, N).
    rand_indices = np.random.randint(0, n, k)
    
    # Keep centroids as dense format, as many entries will be nonzero due to averaging.
    # As long as at least one document in a cluster contains a word,
    # it will carry a nonzero weight in the TF-IDF vector of the centroid.

    centroids = data[rand_indices,:].toarray()
    
    return centroids

 def assign_clusters(data, centroids):
    
    # distances between each data point and the set of centroids:
    distances_from_centroids = pairwise_distances(data, centroids, metric='euclidean')
    
    # cluster assignments for each data point:
    cluster_assignment = np.argmin(distances_from_centroids, axis=1)
    
    return cluster_assignment

def revise_centroids(data, k, cluster_assignment):
    new_centroids = []
    for i in range(k):
        # all data points that belong to cluster i
        member_data_points = data[cluster_assignment==i]
        # the mean of the data points in a cluster
        centroid = member_data_points.mean(axis=0)
        
        # Obtain a flattened ndarray
        centroid = centroid.A1
        new_centroids.append(centroid)
    new_centroids = np.array(new_centroids)
    
    return new_centroids

def compute_heterogeneity(data, k, centroids, cluster_assignment):
    
    heterogeneity = 0.0
    for i in range(k):
        member_data_points = data[cluster_assignment==i, :]
        
        if member_data_points.shape[0] > 0: # check if i-th cluster is non-empty
            # distances from centroid to data points
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances**2
            heterogeneity += np.sum(squared_distances)
        
    return heterogeneity

# Plotting utils
def plot_heterogeneity(heterogeneity, k):
    plt.figure(figsize=(15,8))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Heterogeneity')
    plt.title('Heterogeneity of clustering over time, K={0:d}'.format(k))
    plt.rcParams.update({'font.size': 16})
    plt.show()

def plot_k_vs_heterogeneity(k_values, heterogeneity_values):
    plt.figure(figsize=(15,8))
    plt.plot(k_values, heterogeneity_values, linewidth=4)
    plt.xlabel('K')
    plt.ylabel('Heterogeneity')
    plt.title('K vs. Heterogeneity')
    plt.rcParams.update({'font.size': 16})
    plt.show()

def visualize_document_clusters(raw_docs, tf_idf, centroids, 
	cluster_assignment, k, map_index_to_word, display_content=True):
    '''
	Visualization function in case K-Means is used to cluster textual documents.
	Good clusters should exhibit similarity in topical content.

    raw_docs: original dataframe/SFrame 
    	with a 'text' column for raw documents
    	and a 'name' column for document name/ID
    tf_idf: tf-idf values, sparse matrix format
    map_index_to_word: dataframe/SFrame specifying the mapping betweeen words and column indices
    display_content: if True, display 8 nearest neighbors of each centroid
    '''
    
    print('==========================================================')

    # Visualize each cluster c
    for c in range(k):
        # Cluster heading
        print('Cluster {0:d}    '.format(c)),
        # Print top 5 words with largest TF-IDF weights in the cluster
        idx = centroids[c].argsort()[::-1]
        for i in range(5): # Print each word along with the TF-IDF weight
            print('{0:s}:{1:.3f}'.format(map_index_to_word['category'][idx[i]], centroids[c,idx[i]])),
        print('')
        
        if display_content:
            # Compute distances from the centroid to all data points in the cluster,
            # and compute nearest neighbors of the centroids within the cluster.
            distances = pairwise_distances(tf_idf, [centroids[c]], metric='euclidean').flatten()
            distances[cluster_assignment!=c] = float('inf') # remove non-members from consideration
            nearest_neighbors = distances.argsort()
            # For 8 nearest neighbors, print the title as well as first 180 characters of text.
            # Wrap the text at 80-character mark.
            for i in range(8):
                text = ' '.join(raw_docs[nearest_neighbors[i]]['text'].split(None, 25)[0:25])
                print('\n* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.format(raw_docs[nearest_neighbors[i]]['name'],
                    distances[nearest_neighbors[i]], text[:90], text[90:180] if len(text) > 90 else ''))
        print('==========================================================')





