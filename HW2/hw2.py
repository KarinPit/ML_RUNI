import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    labels = data[:, -1] # Extract labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    gini = 1.0
    for count in counts:
        prob = count / total
        gini -= prob ** 2 # Subtract square of probability
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    labels = data[:, -1]  # Extract labels (last column)
    unique_labels, counts = np.unique(labels, return_counts=True) # Finds all the unique label values & tells how many times each label appears
    total = len(labels)
    entropy = 0.0
    for count in counts:
        prob = count / total
        if prob > 0:
            entropy -= prob * np.log2(prob) # Apply entropy formula
    return entropy

class DecisionNode:
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        labels = self.data[:, -1]
        values, counts = np.unique(labels, return_counts=True)
        pred = values[np.argmax(counts)] # Choose label with highest frequency
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        ## Formula Used: ((number of samples at this node) / (total number of samples at the root)) * (impurity reduction (gain) from splitting on feature A)
        self.feature_importance += self.impurity_func(self.data) * len(self.data) / n_total_sample
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        ## Formula Used: (impurity before split) - sum over feature values[((the number of samples in the subset where feature A has value v) / (the total number of samples at the current node (before splitting)) * (impurity of each subset)]
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        
        feature_values = self.data[:, feature]
        for value in np.unique(feature_values):
            groups[value] = self.data[feature_values == value] # Group by feature value

        impurity_before = self.impurity_func(self.data)
        impurity_after = 0
        for subset in groups.values():
            impurity_after += len(subset) / len(self.data) * self.impurity_func(subset)

        goodness = impurity_before - impurity_after # Information Gain

        if self.gain_ratio:
            split_info = -np.sum([len(subset) / len(self.data) * np.log2(len(subset) / len(self.data)) for subset in groups.values()])
            if split_info != 0:
                goodness /= split_info # Adjust by split information

        return goodness, groups
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        # Check stopping conditions: maximum depth reached OR node is pure (only one class label)
        if self.depth >= self.max_depth or len(np.unique(self.data[:, -1])) == 1:
            self.terminal = True
            return

        best_goodness = -1 # Best goodness of split found so far
        best_feature = None # Best feature to split by
        best_groups = None # Best groups after split

        # Try splitting on each feature
        for feature in range(self.data.shape[1] - 1):
            goodness, groups = self.goodness_of_split(feature)
            if goodness > best_goodness:
                best_goodness = goodness
                best_feature = feature
                best_groups = groups

         # If no good split was found
        if best_goodness <= 0:
            self.terminal = True
            return
        
        # Chi-squared pruning check
        if self.chi < 1:
            dof = len(best_groups) - 1
            chi_square = 0
            labels = self.data[:, -1]
            label_values, label_counts = np.unique(labels, return_counts=True)
            expected = label_counts / len(self.data)

            for subset in best_groups.values():
                subset_labels = subset[:, -1]
                subset_label_values, subset_label_counts = np.unique(subset_labels, return_counts=True)
                for val, count in zip(subset_label_values, subset_label_counts):
                    idx = np.where(label_values == val)[0][0]
                    chi_square += (count - len(subset) * expected[idx]) ** 2 / (len(subset) * expected[idx])

            if chi_square < chi_table[dof][self.chi]:
                self.terminal = True
                return
            
        # Perform split
        self.feature = best_feature
        for value, subset in best_groups.items():
            child = DecisionNode(subset, self.impurity_func, depth=self.depth + 1, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            child.split()
            self.add_child(child, value)

                    
class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        self.root = DecisionNode(self.data, self.impurity_func, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
        self.root.split() # Start splitting recursively from root

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        node = self.root
        # Traverse down the tree following feature values
        while not node.terminal:
            if instance[node.feature] in node.children_values:
                idx = node.children_values.index(instance[node.feature])
                node = node.children[idx]
            else:
                break # If feature value not found in children, stop
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        correct = 0
        for instance in dataset:
            if self.predict(instance) == instance[-1]: # Compare prediction to true label
                correct += 1 
        return correct / len(dataset) * 100 # Accuracy in %
        
    def depth(self):
        return self.root.depth

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = DecisionTree(X_train, calc_entropy, max_depth=max_depth, gain_ratio=True)
        tree.build_tree()
        training.append(tree.calc_accuracy(X_train))
        validation.append(tree.calc_accuracy(X_validation))

    return training, validation


def chi_pruning(X_train, X_validation):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc = []
    depth = []
    chi_values = [0.5, 0.25, 0.1, 0.05, 0.0001] # Different chi levels to test

    for chi in chi_values:
        tree = DecisionTree(X_train, calc_entropy, chi=chi, gain_ratio=True)
        tree.build_tree()
        chi_training_acc.append(tree.calc_accuracy(X_train))
        chi_validation_acc.append(tree.calc_accuracy(X_validation))
        depth.append(tree.root.depth if hasattr(tree.root, 'depth') else 0) # Store tree depth

    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    if node is None:
        return 0
    n_nodes = 1 # Count the current node
    for child in node.children:
        n_nodes += count_nodes(child) # Count children recursively
    return n_nodes






