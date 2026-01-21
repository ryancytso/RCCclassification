import numpy as np
from collections import Counter
from treenode import TreeNode

class DecisionTree():
    """
    Decision Tree Classifier with Rule-Based Feature Gating for RCC Subtype Classification
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features
    """

    def __init__(self, 
                 max_depth=4, 
                 min_samples_leaf=1, 
                 min_information_gain=0.0, 
                 numb_of_features_splitting=None,
                 amount_of_say=None,
                 morphological_depth=None,
                 morphological_features=None,
                 ihc_feature_map=None) -> None:
        """
        Setting the class with hyperparameters
        max_depth: (int) -> max depth of the tree
        min_samples_leaf: (int) -> min # of samples required to be in a leaf to make the splitting possible
        min_information_gain: (float) -> min information gain required to make the splitting possible
        num_of_features_splitting: (str) ->  when splitting if sqrt then sqrt(# of features) features considered, 
                                                            if log then log(# of features) features considered
                                                            else all features are considered
        amount_of_say: (float) -> used for Adaboost algorithm
        morphological_depth: (int) -> depth until which only morphological features are considered
        morphological_features: (list) -> indices of morphological features
        ihc_feature_map: (dict) -> mapping of label to list of IHC feature indices
                                   Example: {0: [10, 11, 12, 13, 14], 1: [15, 16, 17, 18, 19], ...}
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.amount_of_say = amount_of_say
        
        # Feature gating parameters
        self.morphological_depth = morphological_depth
        self.morphological_features = morphological_features if morphological_features is not None else []
        self.ihc_feature_map = ihc_feature_map if ihc_feature_map is not None else {}
        self.use_feature_gating = morphological_depth is not None and len(self.morphological_features) > 0

    def _entropy(self, class_probabilities: list) -> float:
        return sum([-p * np.log2(p) for p in class_probabilities if p>0])
    
    def _class_probabilities(self, labels: list) -> list:
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]

    def _data_entropy(self, labels: list) -> float:
        return self._entropy(self._class_probabilities(labels))
    
    def _partition_entropy(self, subsets: list) -> float:
        """subsets = list of label lists (EX: [[1,0,0], [1,1,1])"""
        total_count = sum([len(subset) for subset in subsets])
        return sum([self._data_entropy(subset) * (len(subset) / total_count) for subset in subsets])
    
    def _split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:
        
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]

        return group1, group2
    
    def _get_top_two_labels(self, data: np.array) -> tuple:
        """
        Returns the two most common labels in the data
        Returns: (most_common_label, second_most_common_label)
        """
        labels = data[:, -1].astype(int)
        label_counts = Counter(labels)
        
        if len(label_counts) == 0:
            return None, None
        elif len(label_counts) == 1:
            most_common = label_counts.most_common(1)[0][0]
            return most_common, None
        else:
            top_two = label_counts.most_common(2)
            return top_two[0][0], top_two[1][0]
    
    def _get_allowed_features(self, data: np.array, current_depth: int) -> list:
        """
        Determines which features are allowed for splitting based on depth and label distribution
        
        For depths <= morphological_depth: only morphological features
        For depths > morphological_depth: IHC features based on top 2 label majorities
        """
        if not self.use_feature_gating:
            # No feature gating, return all features
            return list(range(data.shape[1] - 1))
        
        # Use only morphological features for initial depth
        if current_depth <= self.morphological_depth:
            return self.morphological_features
        
        # After morphological depth, use IHC features based on label distribution
        label1, label2 = self._get_top_two_labels(data)
        
        allowed_ihc_features = []
        if label1 is not None and label1 in self.ihc_feature_map:
            allowed_ihc_features.extend(self.ihc_feature_map[label1])
        if label2 is not None and label2 in self.ihc_feature_map:
            allowed_ihc_features.extend(self.ihc_feature_map[label2])
        
        # Remove duplicates and return
        return list(set(allowed_ihc_features))
    
    def _select_features_to_use(self, data: np.array, current_depth: int) -> list:
        """
        Randomly selects the features to use while splitting w.r.t. hyperparameter numb_of_features_splitting
        and feature gating rules
        """
        # Get allowed features based on gating rules
        feature_idx = self._get_allowed_features(data, current_depth)
        
        if len(feature_idx) == 0:
            # No features available, return empty list
            return []

        if self.numb_of_features_splitting == "sqrt":
            num_features = max(1, int(np.sqrt(len(feature_idx))))
            feature_idx_to_use = np.random.choice(feature_idx, size=num_features, replace=False)
        elif self.numb_of_features_splitting == "log":
            num_features = max(1, int(np.log2(len(feature_idx))))
            feature_idx_to_use = np.random.choice(feature_idx, size=num_features, replace=False)
        else:
            feature_idx_to_use = feature_idx

        return list(feature_idx_to_use)
        
    def _find_best_split(self, data: np.array, current_depth: int) -> tuple:
        """
        Finds the best split (with the lowest entropy) given data
        Returns 2 splitted groups and split information
        """
        min_part_entropy = 1e9
        feature_idx_to_use = self._select_features_to_use(data, current_depth)
        
        if len(feature_idx_to_use) == 0:
            # No features to split on, return data as-is
            return data, np.array([]), -1, -1, 1e9

        for idx in feature_idx_to_use:
            feature_vals = np.percentile(data[:, idx], q=np.arange(25, 100, 25))
            for feature_val in feature_vals:
                g1, g2, = self._split(data, idx, feature_val)
                
                # Skip if either group is empty
                if len(g1) == 0 or len(g2) == 0:
                    continue
                    
                part_entropy = self._partition_entropy([g1[:, -1], g2[:, -1]])
                if part_entropy < min_part_entropy:
                    min_part_entropy = part_entropy
                    min_entropy_feature_idx = idx
                    min_entropy_feature_val = feature_val
                    g1_min, g2_min = g1, g2

        # Check if we found a valid split
        if min_part_entropy == 1e9:
            return data, np.array([]), -1, -1, 1e9
            
        return g1_min, g2_min, min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy

    def _find_label_probs(self, data: np.array) -> np.array:

        labels_as_integers = data[:,-1].astype(int)
        # Calculate the total number of labels
        total_labels = len(labels_as_integers)
        # Calculate the ratios (probabilities) for each label
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)

        # Populate the label_probabilities array based on the specific labels
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities

    def _create_tree(self, data: np.array, current_depth: int) -> TreeNode:
        """
        Recursive, depth first tree creation algorithm with feature gating
        """

        # Check if the max depth has been reached (stopping criteria)
        if current_depth > self.max_depth:
            return None
        
        # Find best split (now considering feature gating)
        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self._find_best_split(data, current_depth)
        
        # Find label probs for the node
        label_probabilities = self._find_label_probs(data)

        # Calculate information gain
        node_entropy = self._entropy(label_probabilities)
        information_gain = node_entropy - split_entropy
        
        # Create node
        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)

        # Check if no valid split was found
        if split_feature_idx == -1 or len(split_2_data) == 0:
            return node
        
        # Check if the min_samples_leaf has been satisfied (stopping criteria)
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node
        # Check if the min_information_gain has been satisfied (stopping criteria)
        elif information_gain < self.min_information_gain:
            return node

        current_depth += 1
        node.left = self._create_tree(split_1_data, current_depth)
        node.right = self._create_tree(split_2_data, current_depth)
        
        return node
    
    def _predict_one_sample(self, X: np.array) -> np.array:
        """Returns prediction for 1 dim array"""
        node = self.tree

        # Finds the leaf which X belongs
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """
        Trains the model with given X and Y datasets
        """
        # Build label encodings so it works with strings or ints
        unique = np.unique(Y_train)
        self.label_to_int = {lab: i for i, lab in enumerate(unique)}
        self.int_to_label = {i: lab for lab, i in self.label_to_int.items()}

        Y_enc = np.array([self.label_to_int[y] for y in Y_train], dtype=int)

        # Concat features and labels
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, Y_enc.reshape((-1, 1))), axis=1)

        # Start creating the tree
        self.tree = self._create_tree(data=train_data, current_depth=0)

        # Calculate feature importance
        self.feature_importances = dict.fromkeys(range(X_train.shape[1]), 0)
        self._calculate_feature_importance(self.tree)

        # Normalize the feature importance values
        total_importance = sum(self.feature_importances.values())
        if total_importance > 0:
            self.feature_importances = {k: v / total_importance for k, v in self.feature_importances.items()}

    def predict_proba(self, X_set: np.array) -> np.array:
        """Returns the predicted probs for a given data set"""

        pred_probs = np.apply_along_axis(self._predict_one_sample, 1, X_set)
        
        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """Returns the predicted labels for a given data set"""
        pred_int = np.argmax(self.predict_proba(X_set), axis=1)

        # if mapping exists, return original string labels
        if hasattr(self, "int_to_label"):
            return np.array([self.int_to_label[i] for i in pred_int], dtype=object)
        
        return pred_int 
        
    def _print_recursive(self, node: TreeNode, level=0) -> None:
        if node != None:
            self._print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' + node.node_def())
            self._print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        self._print_recursive(node=self.tree)

    def _calculate_feature_importance(self, node):
        """Calculates the feature importance by visiting each node in the tree recursively"""
        if node != None:
            self.feature_importances[node.feature_idx] += node.feature_importance
            self._calculate_feature_importance(node.left)
            self._calculate_feature_importance(node.right)
    
    def get_feature_usage_report(self) -> dict:
        """
        Returns a report showing which features were used at each depth level
        Useful for debugging feature gating behavior
        """
        report = {}
        self._collect_feature_usage(self.tree, 0, report)
        return report
    
    def _collect_feature_usage(self, node, depth, report):
        """Helper function to collect feature usage statistics"""
        if node is None:
            return
        
        if depth not in report:
            report[depth] = []
        
        if node.feature_idx >= 0:  # Valid feature
            report[depth].append(node.feature_idx)
        
        if node.left:
            self._collect_feature_usage(node.left, depth + 1, report)
        if node.right:
            self._collect_feature_usage(node.right, depth + 1, report)