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
        Works with both categorical and integer labels
        """
        labels = data[:, -1]
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
        # Get labels from data (could be strings or integers)
        labels = data[:, -1]
        
        # Calculate the total number of labels
        total_labels = len(labels)
        # Calculate the ratios (probabilities) for each label
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)

        # Populate the label_probabilities array based on the specific labels
        for i, label in enumerate(self.labels_in_train):
            # Count occurrences of this label
            label_count = np.sum(labels == label)
            if label_count > 0:
                label_probabilities[i] = label_count / total_labels

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
        Handles both categorical (string) and integer labels
        """
        # Store unique labels and create label mapping
        self.labels_in_train = np.unique(Y_train)
        
        # Create a mapping from original labels to integer indices
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels_in_train)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Concat features and labels (keep original label format)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)

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
        """Returns the predicted labels for a given data set in original label format"""

        pred_probs = self.predict_proba(X_set)
        pred_indices = np.argmax(pred_probs, axis=1)
        
        # Convert indices back to original labels
        preds = np.array([self.labels_in_train[idx] for idx in pred_indices])
        
        return preds    
        
    def _print_recursive(self, node: TreeNode, level=0) -> None:
        if node != None:
            self._print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' + node.node_def())
            self._print_recursive(node.right, level + 1)

    #def print_tree(self) -> None:
        #self._print_recursive(node=self.tree)

    def print_tree(self, feature_names=None) -> None:
        """
        Print the decision tree in a readable format
        
        Parameters:
        -----------
        feature_names : list, optional
            List of feature names for better readability
        """
        if feature_names is not None:
            self.feature_names = feature_names
            
        else:
            self.feature_names = None
            
        print("\n" + "="*80)
        print("DECISION TREE STRUCTURE")
        print("="*80 + "\n")
        self._print_recursive(node=self.tree)
        print("\n" + "="*80 + "\n")

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

    # End of actual tree features, below features are for visualization and analysis
    def get_tree_summary(self) -> dict:
        """
        Returns a comprehensive summary of the tree including:
        - Tree depth
        - Number of leaves
        - Feature usage statistics
        - Class distribution at leaves
        """
        summary = {
            'max_depth_reached': 0,
            'num_leaves': 0,
            'num_internal_nodes': 0,
            'leaf_class_distributions': [],
            'feature_usage_count': {},
            'samples_per_depth': {}
        }
        
        self._analyze_tree(self.tree, 0, summary)
        
        return summary
    
    def _analyze_tree(self, node, depth, summary):
        """Helper function to analyze tree structure"""
        if node is None:
            return
        
        # Update max depth
        summary['max_depth_reached'] = max(summary['max_depth_reached'], depth)
        
        # Track samples per depth
        if depth not in summary['samples_per_depth']:
            summary['samples_per_depth'][depth] = 0
        summary['samples_per_depth'][depth] += len(node.data)
        
        # Check if leaf
        is_leaf = (node.left is None and node.right is None)
        
        if is_leaf:
            summary['num_leaves'] += 1
            # Store leaf information
            pred_class_idx = np.argmax(node.prediction_probs)
            pred_class = self.labels_in_train[pred_class_idx]
            summary['leaf_class_distributions'].append({
                'predicted_class': pred_class,
                'probabilities': dict(zip(self.labels_in_train, node.prediction_probs)),
                'num_samples': len(node.data)
            })
        else:
            summary['num_internal_nodes'] += 1
            # Track feature usage
            if node.feature_idx >= 0:
                if node.feature_idx not in summary['feature_usage_count']:
                    summary['feature_usage_count'][node.feature_idx] = 0
                summary['feature_usage_count'][node.feature_idx] += 1
        
        # Recurse on children
        if node.left:
            self._analyze_tree(node.left, depth + 1, summary)
        if node.right:
            self._analyze_tree(node.right, depth + 1, summary)
    
    def print_summary(self, feature_names=None):
        """
        Print a comprehensive summary of the decision tree
        """
        summary = self.get_tree_summary()
        
        print("\n" + "="*80)
        print("DECISION TREE SUMMARY")
        print("="*80)
        print(f"\nTree Depth: {summary['max_depth_reached']}")
        print(f"Number of Leaves: {summary['num_leaves']}")
        print(f"Number of Internal Nodes: {summary['num_internal_nodes']}")
        
        print("\n" + "-"*80)
        print("FEATURE IMPORTANCE (normalized)")
        print("-"*80)
        sorted_features = sorted(self.feature_importances.items(), 
                                key=lambda x: x[1], reverse=True)
        for feat_idx, importance in sorted_features[:10]:  # Top 10
            if importance > 0:
                feat_name = f"Feature[{feat_idx}]"
                if feature_names and feat_idx < len(feature_names):
                    feat_name = feature_names[feat_idx]
                print(f"  {feat_name:30s}: {'█' * int(importance * 50)}{importance:.4f}")
        
        print("\n" + "-"*80)
        print("FEATURE USAGE IN TREE")
        print("-"*80)
        sorted_usage = sorted(summary['feature_usage_count'].items(), 
                             key=lambda x: x[1], reverse=True)
        for feat_idx, count in sorted_usage[:10]:  # Top 10
            feat_name = f"Feature[{feat_idx}]"
            if feature_names and feat_idx < len(feature_names):
                feat_name = feature_names[feat_idx]
            print(f"  {feat_name:30s}: Used {count} times")
        
        print("\n" + "-"*80)
        print("LEAF NODE CLASS DISTRIBUTIONS")
        print("-"*80)
        class_counts = {}
        for leaf in summary['leaf_class_distributions']:
            pred_class = leaf['predicted_class']
            if pred_class not in class_counts:
                class_counts[pred_class] = 0
            class_counts[pred_class] += 1
        
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name:30s}: {count} leaf nodes")
        
        if self.use_feature_gating:
            print("\n" + "-"*80)
            print("FEATURE GATING INFORMATION")
            print("-"*80)
            print(f"  Morphological Depth: {self.morphological_depth}")
            print(f"  Morphological Features: {len(self.morphological_features)} features")
            print(f"  IHC Feature Map: {len(self.ihc_feature_map)} subtypes configured")
            
            usage_report = self.get_feature_usage_report()
            print("\n  Feature Usage by Depth:")
            for depth in sorted(usage_report.keys()):
                features_used = set(usage_report[depth])
                print(f"    Depth {depth}: {len(features_used)} unique features used")
                if depth <= self.morphological_depth:
                    print(f"              (Morphological phase)")
                else:
                    print(f"              (IHC phase)")
        
        print("\n" + "="*80 + "\n")