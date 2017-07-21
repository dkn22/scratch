'''
email: dat.nguyen at cantab.net
'''

import pandas as pd
import json
import numpy as np

class Btree_Model(object):
    '''
    An implementation of binary decision trees. Works only for binary features.
    An instance of the class is a model - in order to create the tree, use the create method, 
    which returns the tree as a recursive dictionary. Hence, the instance (model) can be used to create several trees.
    Only the last tree is stored as an attribute.
    
    The model is equipped with classification and node count methods, but a tree must be passed to each.
    '''
    def __init__(self, max_depth=10, min_node_size=1, 
                 min_error_reduction=0.0):
        self.max_depth = max_depth
        self.tree = None
        self.min_node_size = min_node_size
        self.min_error_reduction = min_error_reduction
        print('A binary decision tree of max depth %d is initiated.' \
              %(self.max_depth))
        print('No further splits are allowed if a node has fewer than %d data points' \
             %self.min_node_size)
        print('As a stopping condition, we require that each split generate error reduction of at least %.1f' \
             %(self.min_error_reduction))
    
    @staticmethod
    def intermediate_node_num_mistakes(labels_in_node):
        if len(labels_in_node) == 0: return 0  
        labels_in_node = np.array(labels_in_node)
        num_positive = np.sum(labels_in_node==1)
        num_negative = np.sum(labels_in_node==-1)
        
        # Classification is by the majority rule, hence the minority class encompasses all mistakes
        num_mistakes = min(num_positive, num_negative)
        return num_mistakes
    
    @staticmethod
    def best_splitting_feature(data, features, target):
        assert type(features)==list and type(target)==str, "Features must be a list, and target must be a string."
        target_values = data[target]
        best_feature = None # Keep track of the best feature 
        best_error = 10     # Keep track of the best error so far
        
        # num_data_points = float(len(data)) # float division not an issue in Python 3
        
        for feature in features:
            # 1. Split on the binary feature
            left_split = data[data[feature] == 0]
            right_split = data[data[feature] == 1]
            
            # 2. Calculate the number of misclassified examples
            left_mistakes = Btree_Model.intermediate_node_num_mistakes(left_split[target])
            right_mistakes = Btree_Model.intermediate_node_num_mistakes(right_split[target])
            
            # 3. Calculate classification error for the split on this feature
            error = (left_mistakes + right_mistakes) / data.shape[0]
            
            # 4. Track the best feature
            if error <= best_error:
                best_feature = feature
                best_error = error
        return best_feature
        
    @staticmethod
    def create_leaf(target_values):
        '''
        Each node in the decision tree is represented as a dictionary.
        '''
        leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True}
        
        num_ones = len(target_values[target_values == +1])
        num_minus_ones = len(target_values[target_values == -1])
        
        if num_ones > num_minus_ones:
            leaf['prediction'] = +1
        else:
            leaf['prediction'] = -1
        
        return leaf
    
    @staticmethod
    def error_reduction(error_before_split, error_after_split):
        '''
        computes the gain in error following a node split.
        '''
        return error_before_split - error_after_split
    
    def reached_minimum_node_size(self, data):
        return len(data) <= self.min_node_size
    
    def create(self, data, features, target, current_depth = 0):
        '''
        A recursive greedy algorithm to build the binary decision tree. 
        Max depth is stored as an attribute of the instance.
        '''
        self.target = target
        self.features = features
        remaining_features = list(features)
        target_values = data[target]
        print("--------------------------------------------------------------------")
        print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))
        
        # A recursive algorithm, hence we define stopping conditions
        
        if Btree_Model.intermediate_node_num_mistakes(target_values) == 0:
            print("Stopping condition 1 (single class node) reached.")     
            # If no mistakes at current node, make current node a leaf node
            return Btree_Model.create_leaf(target_values)
        
        if len(remaining_features) == 0:
            print("Stopping condition 2 (no more features) reached.")     
            return Btree_Model.create_leaf(target_values)
        
        if current_depth >= self.max_depth:
            print("Stopping condition 3 (max depth) reached.")     
            return Btree_Model.create_leaf(target_values)
        
        if self.reached_minimum_node_size(target_values):       
            print("Early stopping condition reached. Reached minimum node size.")
            return Btree_Model.create_leaf(target_values)
        
        # Determine best feature to split on
        splitting_feature = Btree_Model.best_splitting_feature(data, 
                                        remaining_features, target)
        
        # Split the node
        left_split = data[data[splitting_feature] == 0]
        right_split = data[data[splitting_feature] == 1]
        remaining_features.remove(splitting_feature)
        print("Split on feature %s. (%d, %d)" % (\
                      splitting_feature, len(left_split), len(right_split)))
        
        # Another stopping condition: don't split if insufficient error reduction
        error_before_split = Btree_Model.intermediate_node_num_mistakes(target_values) / len(data)
    
        left_mistakes = Btree_Model.intermediate_node_num_mistakes(left_split[target])
        right_mistakes = Btree_Model.intermediate_node_num_mistakes(right_split[target])
        error_after_split = (left_mistakes + right_mistakes) / len(data)
    
        if Btree_Model.error_reduction(error_before_split, 
                                       error_after_split) <= self.min_error_reduction: 
            print("Early stopping condition reached. Minimum error reduction.")
            return Btree_Model.create_leaf(target_values)
        
        # If a split generates an empty node (i.e., all data goes to either left or right)
        if len(left_split) == len(data):
            print("Creating leaf node.")
            return Btree_Model.create_leaf(left_split[target])
        if len(right_split) == len(data):
            print("Creating leaf node.")
            return Btree_Model.create_leaf(right_split[target])
        
        left_tree = self.create(left_split, remaining_features, 
                                target, current_depth+1)
        right_tree = self.create(right_split, remaining_features, 
                                 target, current_depth+1)
        
        self.tree = {'is_leaf'          : False, 
                    'prediction'       : None,
                    'splitting_feature': splitting_feature,
                    'left'             : left_tree, 
                    'right'            : right_tree}
        
        
        return self.tree
    
    def classify(self, tree, obs, annotate=False):
        '''
        Classifies an observation (array-like of inputs) by traversing the tree.
        '''
        if tree['is_leaf']:
            if annotate:
                print("At leaf, predicting %s" %(tree['prediction']))
            return tree['prediction']
        else:
            split_feature_value = obs[tree['splitting_feature']]
            if annotate:
                print("Split on %s = %s" % (tree['splitting_feature'], 
                                            split_feature_value))
            if split_feature_value == 0:
                return self.classify(tree['left'], obs, annotate)
            else:
                return self.classify(tree['right'], obs, annotate)
    
    def count_nodes(self, tree):
        if tree['is_leaf']:
            return 1
        return 1 + self.count_nodes(tree['left']) + self.count_nodes(tree['right'])
    
    def evaluate(self, tree, data):
        # calcuates classification error on data
        predictions = data.apply(lambda x: self.classify(tree, x), axis=1)
        accuracy = np.sum(predictions==data[self.target])/data.shape[0]
        return 1-accuracy