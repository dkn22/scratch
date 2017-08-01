'''
Approximate nearest neighbor search: locality sensitive hashing.

Work in progress.
'''

import numpy as np
import pandas as pd
from utils import Distances
from itertools import combinations
from copy import copy
# from sklearn.metrics.pairwise import pairwise_distances

class NearestNeighborsLSH(Distances):

    def __init__(self, metric='cosine'):
        Distances.__init__(self, metric)
        self._model = None

    def _gen_random_vectors(self, num_vectors, dim):

        return np.random.randn(dim, num_vectors)

    def train(self, data, num_vector=16, seed=None):
    
        dim = data.shape[1]
        if seed is not None:
            np.random.seed(seed)
        random_vectors = self._gen_random_vectors(num_vector, dim)
      
        powers_of_two = (1 << np.arange(num_vector-1, -1, -1))
      
        table = {}
        
        # Partition data points into bins
        bin_index_bits = (data.dot(random_vectors) >= 0)
      
        # Encode bin index bits into integers
        bin_indices = bin_index_bits.dot(powers_of_two)
        
        # Update `table` so that `table[i]` is the list of ids with bin index equal to i.
        for data_index, bin_index in enumerate(bin_indices):
            if bin_index not in table:
                table[bin_index] = list()
            
            # Fetch the list of ids associated with the bin and add to the end.
            table[bin_index].append(data_index)

        self._model = {'data': data,
                      'bin_index_bits': bin_index_bits,
                      'bin_indices': bin_indices,
                      'table': table,
                      'random_vectors': random_vectors
                      }

        print('Training completed.')
        
        # return model

    @classmethod
    def search_nearby_bins(cls, query_bin_bits, table, search_radius=2, 
                           initial_candidates=set()):
        """
        For a given query vector and trained LSH model, return all candidate neighbors for
        the query among all bins within the given search radius.
        
        In our case, returns the set of IDs that we would search the nearest neighbours from.
        
        Example usage
        -------------
        >>> LSH = NearestNeighborsLSH(metric='cosine')
        >>> LSH.train(data, num_vector=16, seed=143)
        >>> q = LSH.model['bin_index_bits'][0]  # vector for the first obs
      
        >>> candidates = LSH.search_nearby_bins(q, model['table'])
        """
        num_vector = len(query_bin_bits)
        powers_of_two = 1 << np.arange(num_vector-1, -1, -1)
        
        # Allow the user to provide an initial set of candidates.
        candidate_set = initial_candidates[:]
        
        for different_bits in combinations(range(num_vector), search_radius):       
            # Flip the bits (n_1,n_2,...,n_r) of the query bin to produce a new bit vector.
            
            alternate_bits = query_bin_bits[:]
            for i in different_bits:
                alternate_bits[i] = not alternate_bits[i]
            
            # Convert the new bit vector to an integer index
            nearby_bin = alternate_bits.dot(powers_of_two)
            
            # Fetch the list of obs belonging to the bin indexed by the new bit vector.
            # Then add those obs to candidate_set
            if nearby_bin in table:
                candidate_set.update(table[nearby_bin])
                
        return candidate_set

    def query(self, vec, k, max_search_radius=1):

        assert self._model is not None

        model = self._model

        try:
            if len(vec.shape) == 1:
                vec = vec[:, np.newaxis]
        except Exception:
            pass

        data = model['data']
        table = model['table']
        random_vectors = model['random_vectors']
        num_vector = random_vectors.shape[1]
        
        
        # bin index for the query vector, in bit representation.
        bin_index_bits = (vec.dot(random_vectors) >= 0).flatten()
        
        candidate_set = set()
        for search_radius in range(max_search_radius+1):
            candidate_set = self.search_nearby_bins(bin_index_bits, table, search_radius, initial_candidates=candidate_set)
        
        nearest_neighbors = pd.DataFrame.from_dict({'id':candidate_set})
        candidates = data[np.array(list(candidate_set)), :]

        distances = []
        for idx in len(candidates):
            try:
                distances.append(self.compute(vec, candidates.iloc[idx, :]))
            except AttributeError:
                distances.append(self.compute(vec, candidates[idx, :]))

        nearest_neighbors['distance'] = distances
        # nearest_neighbors['distance'] = pairwise_distances(candidates, vec, metric=self.metric).flatten()
        
        k_nearest = nearest_neighbors.sort_values(by=['distance'], ascending=False)[:k]

        return k_nearest

