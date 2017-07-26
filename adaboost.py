'''
An implementation of the AdaBoost algorithm.

The algorithm is only suitable for categorical (if one-hot encoded)
or binary features.

Thanks to the UofWashington ML course (Coursera), on which
this implementation is based.

email: dat.nguyen at cantab.net
'''

import math
import numpy as np
import pandas as pd

# one-hot encoding categorical features

def one_hot_encode(df, features=None, **kwargs):
    '''
    df: original dataframe
    features: list of columns to one-hot encode

    Returns a new dataframe with encoded variables.
    The original columns are dropped.
    '''
    new_df = pd.get_dummies(df, columns=features, **kwargs)
    new_df = new_df.drop(features, axis=1)

    return new_df



class AdaBoostBinary(object):
    '''
    Adaboost algorithm for binary features.
    '''
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.weights = None
        self.tree_stumps = None

    def fit(self, df, features, target,
            num_trees, verbose=False):
        '''
        df: dataframe/SFrame
        features: list of columns in the df
        target: target column name in the df
        num_trees: number of trees/stumps to learn

        '''
        assert isinstance(target, str)

        alpha = np.array([1.]*len(df))
        weights = []
        tree_stumps = []
        target_values = df[target]

        for t in range(num_trees):
            if verbose:
                print('=====================================================')
                print('Adaboost Iteration %d' % t)
                print('=====================================================')
            
            tree_stump = self._weighted_tree(df, features, target,
                                             data_weights=alpha, max_depth=self.max_depth)
            tree_stumps.append(tree_stump)

            predictions = df.apply(lambda x: self._classify(tree_stump, x))

            is_correct = predictions == target_values
            is_wrong   = predictions != target_values
            
            weighted_error = sum(alpha[is_wrong]) / sum(alpha)
            
            # model coefficient using weighted error
            weight = 0.5 * math.log((1-weighted_error)/weighted_error)
            weights.append(weight)
            
            # Adjust weights on data point
            adjustment = is_correct.apply(lambda is_correct: np.exp(-weight) if is_correct else np.exp(weight))
            
            # Boosting
            alpha = alpha*adjustment
            alpha = alpha/sum(alpha)
            
        self.weights = weights 
        self.tree_stumps = tree_stumps

    def _weighted_tree(self, df, features, target,
                       data_weights, current_depth=1, max_depth=10,
                       verbose=False):
        '''
        df: dataframe/SFrame
        features: list of columns in the df
        target: target column name in the df
        data_weights: array-like of weights for each observation
        current_depth: current level of the tree
        max_depth: maximum depth of the tree permitted
        '''

        # data = df
        target_values = df[target]
        remaining_features = list(features[:])

        if verbose:
            print("--------------------------------------------------------------------")
            print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))
        
        # Stopping condition 1: single-class node (error is 0)
        if self._interm_w_error(target_values, data_weights) == 0:
            if verbose: print('Single-class node. Creating a leaf.')
            return self._create_leaf(target_values, data_weights)
        
        # Stopping condition 2: no more features to split on
        if len(remaining_features) < 1:
            if verbose: print('No more features to split on. Creating a leaf.')
            return self._create_leaf(target_values, data_weights)
        
        # Early stopping condition: reached maximum depth
        if current_depth > max_depth:
            if verbose: print('Reached maximum depth. Creating a leaf.')
            return self._create_leaf(target_values, data_weights)
        
        best_feature = self._find_split(df, features, target, data_weights)
        
        left_split = df[df[best_feature] == 0]
        left_weights = data_weights[df[best_feature] == 0]
        
        right_split = df[df[best_feature] == 1]
        right_weights = data_weights[df[best_feature] == 1]
        
        print("Split on feature %s. (%s, %s)" % (\
                  best_feature, len(left_split), len(right_split)))
        
        remaining_features.remove(best_feature)
        
        # Create leaves if the split is trivial, i.e. one of the split nodes is empty
        if len(left_split) == len(df):
            return self._create_leaf(left_split[target], left_weights)
        if len(right_split) == len(df):
            return self._create_leaf(right_split[target], right_weights)
        
        
        # Recursion on sub-trees
        left_subtree = self._weighted_tree(left_split, remaining_features, target, left_weights,
                                                    current_depth+1, max_depth)
        
        right_subtree = self._weighted_tree(right_split, remaining_features, target, right_weights,
                                                     current_depth+1, max_depth)
        
        return {'is_leaf': False,
               'splitting_feature': best_feature,
               'left': left_subtree,
               'right': right_subtree,
               'prediction': None}

    def _interm_w_error(self, labels_in_node, data_weights):

        weighted_error_all_negative = sum(data_weights[labels_in_node == +1])
        
        weighted_error_all_positive = sum(data_weights[labels_in_node == 0])
        
        #  majority rule classification
        find_min = lambda x, y: (x, +1) if x <= y else (y, 0)
        return find_min(weighted_error_all_positive, weighted_error_all_negative)

    def _create_leaf(self, y, data_weights):

        leaf = {'splitting_feature': None,
               'is_leaf': True}
        
        error, class_label = self._interm_w_error(y, data_weights)
        leaf['prediction'] = class_label
        return leaf

    def _find_split(self, df, features, target, data_weights):

        best_feature = None
        best_error = np.inf
        num_points = float(len(df))
            
        for feature in features:
            left_split = df[df[feature] == 0]
            right_split = df[df[feature] == 1]
            
            left_data_weights = data_weights[df[feature] == 0]
            right_data_weights = data_weights[df[feature] == 1]
            
            left_weighted_error, left_class = self._interm_w_error(left_split[target], left_data_weights)
            right_weighted_error, right_class = self._interm_w_error(right_split[target], right_data_weights)
            
            weighted_error = (left_weighted_error + right_weighted_error) / sum(data_weights)
            
            if weighted_error < best_error:
                best_feature = feature
                best_error = weighted_error
            
        return best_feature

    def _compute_error(self, tree, df, target):
        prediction = df.apply(lambda x: self._classify(tree, x))
        
        return (prediction != df[target]).sum() / float(len(df))

    def _classify(self, tree, point, annotate=False):

        if tree['is_leaf']:
            if annotate:
                print('At leaf, predicting %s', tree['prediction'])
            return tree['prediction']
    
        else:
            split_feature_value = point[tree['splitting_feature']]
            if annotate:
                print('Splitting on %s = %s' %(tree['splitting_feature'], split_feature_value))
            
            if split_feature_value == 0:
                return self._classify(tree['left'], point, annotate)
            
            else:
                return self._classify(tree['right'], point, annotate)

    def predict(self, df, stump_weights=None, tree_stumps=None):
        '''
        df: dataframe/SFrame
        stump_weights: list of weights for each tree/stump in the ensemble
        tree_stumps: list of trees/stumps in the ensemble
        '''
        if stump_weights is None:
            stump_weights = self.weights
        if tree_stumps is None:
            tree_stumps = self.tree_stumps

        scores = np.array([0.]*len(df))
        
        for i, tree_stump in enumerate(tree_stumps):
            predictions = df.apply(lambda x: self._classify(tree_stump, x))
            
            # Accumulate predictions on scores array
            scores = scores + stump_weights[i]*predictions
            
        return scores.apply(lambda score: +1 if score > 0 else -1)

    def _count_nodes(self, tree):
        if tree['is_leaf']:
            return 1
        else:
            return 1 + self._count_nodes(tree['left']) + self._count_nodes(tree['right'])







