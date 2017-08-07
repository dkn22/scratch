'''
Implementation of the ID3 decision tree algorithm.

Utilises some of the logic I learnt through Dataquest.

email: dat.nguyen at cantab.net
'''

import numpy as np
import math
import pandas as pd

class ID3tree(object):

	def __init__(self):
		self.tree = {}
		self.nodes = []

	def _calc_entropy(self, col):
		'''
		col: a Series, list, or numpy array.

	    Calculate entropy given a Series, list, or numpy array.
	    '''

	    counts = np.bincount(column) # unique value counts
	    probabilities = counts / len(column)
	    
	    entropy = 0
	    for prob in probabilities:
	        if prob > 0:
	            entropy += prob * math.log(prob, 2)
	    
	    return -entropy

	def _calc_info_gain(self, data, split_name, target):
		'''
		data: dataset in a structured format (e.g., DataFrame)
		split_name: name of the column (feature) to split on
		target: target variable as a string

	    Calculate information gain given a dataset, column to split on, and target.
	    '''
	    assert isinstance(split_name, str)
	    assert isinstance(target, str)

	    original_entropy = self._calc_entropy(data[target])
	    
	    column = data[split_name]
	    # split will occur on the median
	    median = column.median()
	    
	    left_split = data[column <= median]
	    right_split = data[column > median]
	    
	    weighted_entropies = 0
	    for subset in [left_split, right_split]:

	        prob = (subset.shape[0] / data.shape[0]) 
	        weighted_entropies += prob * self._calc_entropy(subset[target])
	    
	    information_gain = original_entropy - weighted_entropies

	    return information_gain

	def _find_split(self, data, target, features):
		'''
		data: dataset in a structured format (e.g., DataFrame)
		target: target variable as a string
		features: list of feature names
		'''
	    info_gains = [self._calc_info_gain(data, x, target) for x in features]
	    best_column = features[np.argmax(info_gains)]
	    return best_column

	def _train(self, data, target, features, tree):
		'''
		data: dataset in a structured format (e.g., DataFrame)
		target: target variable as a string
		features: list of feature names
		tree: a dictionary representation of the tree

		Train a decision tree using the ID3 algorithm.
		'''

		assert isinstance(tree, dict)

		unique_targets = pd.unique(data[target])
		self.nodes.append(len(self.nodes) + 1)

		tree['number'] = self.nodes[-1]

		# if singleton class, create a leaf node
		if len(unique_targets) == 1:
			if unique_targets[0] == 0:
				tree['label'] = 0
			else:
				tree['label'] = 1
			return

		best_column = self._find_split(data, target, features)
		column_median = data[best_column].median()

		tree['split_feature'] = best_column
		tree['median'] = column_median

		left_split = data[data[best_column] <= column_median]
		right_split = data[data[best_column] > column_median]

		split_dict = {'left': left_split, 'right': right_split}

		for key, split in split_dict.items():
			tree[key] = {}
			self.train(split, target, features, tree[key])

	def fit(self, X, y, features=None):
		'''
		X: feature matrix
		y: target variable as array-like
		features: list of features
		'''

		# assert a previous tree has not been trained on this instance
		assert not self.tree

		if features is None:
			features = X.columns

		self.train(X, y, features, self.tree)

	def predict(self, X_test):
		# functional approach
		def _predict_row(tree, row):
			if "label" in tree:
		        return tree["label"]
		    
		    column = tree["column"]
		    median = tree["median"]

		    if row[column] <= median:
		        return _predict_row(tree['left'], row)
		    elif row[column] > median:
		        return _predict_row(tree['right'], row)

		predictions = X_test.apply(lambda x: _predict_row(self.tree, x), axis=1)

		return predictions


# utils

def print_node(tree, depth):
    if "label" in tree:
        print_with_depth("Leaf: Label {0}".format(tree["label"]), depth)
        return

    print_with_depth("{0} > {1}".format(tree["split_feature"], tree["median"]), depth)
    
    branches = [tree["left"], tree["right"]]
        
    for branch in branches:
        print_node(branch, depth+1)

def print_with_depth(string, depth):
    
    prefix = "    " * depth
    print("{0}{1}".format(prefix, string))



