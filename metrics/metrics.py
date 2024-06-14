"""
Authors:
Shawn Fan
Artur Dox
Ang Li
Holly Whitfield
Ramon Vinas
---
Date: June 13, 2024
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.cluster import KMeans
from pertpy.tools._distances._distances import AbstractDistance
import pertpy as pt

def jaccard_similarity(list1, list2):
    """
    Compute the Jaccard similarity between two lists.
    """
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2)) / len(set1.union(set2))

class BhattacharyyaDistance(AbstractDistance):
    # Measures the overlap between two statistical samples
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        # Normalize the means to probability distributions
        X_norm = X_mean / np.sum(X_mean)
        Y_norm = Y_mean / np.sum(Y_mean)
        BC = np.sum(np.sqrt(X_norm * Y_norm))
        return -np.log(BC)

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("Bhattacharyya distance cannot be calculated from precomputed matrix directly.")

class JaccardDistance(AbstractDistance):
    # A measure of dissimilarity based on set comparison (intersection and union)
    # gene expressions are binarized based on a threshold (e.g., gene is expressed or not)
    def __init__(self, threshold=0.5) -> None:
        self.threshold = threshold
        self.accepts_precomputed = True
    
    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        X_binary = X.mean(axis=0) > self.threshold
        Y_binary = Y.mean(axis=0) > self.threshold
        intersection = np.logical_and(X_binary, Y_binary).sum(axis=0)
        union = np.logical_or(X_binary, Y_binary).sum(axis=0)
        # Convert Jaccard Index to distance
        return 1 - np.mean(intersection / union)
    
    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("Jaccard distance cannot be calculated from precomputed matrix directly.")

    
class F1ScoreDistance(AbstractDistance):
    # The harmonic mean of precision and recall
    # gene expressions are binarized based on a threshold (e.g., gene is expressed or not)
    def __init__(self, threshold=0.5) -> None:
        self.threshold = threshold
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        # Binarize the gene expression data based on the threshold
        X_binary = (X.mean(axis=0) > self.threshold).astype(int)
        Y_binary = (Y.mean(axis=0) > self.threshold).astype(int)
        # Calculate F1 Score
        f1 = f1_score(Y_binary.flatten(), X_binary.flatten(), average='macro')
        # 1 - F1 Score to represent distance
        return 1 - f1
    
    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("F1 score cannot be calculated from a pairwise distance matrix.")

class ARIDistance(AbstractDistance):
    # Measures the similarity between two data clusterings, adjusted for chance
    def __init__(self, n_clusters=5) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        X_mean = X.mean(axis=0)
        Y_mean = X.mean(axis=0)
        # Clustering the data
        kmeans_X = KMeans(n_clusters=self.n_clusters, random_state=42).fit(X_mean.reshape(-1, 1))
        kmeans_Y = KMeans(n_clusters=self.n_clusters, random_state=42).fit(Y_mean.reshape(-1, 1))
        # Get the cluster labels from both predictions and ground truth
        labels_X = kmeans_X.labels_
        labels_Y = kmeans_Y.labels_
        # Calculating ARI
        ari_score = adjusted_rand_score(labels_X, labels_Y)
        # Convert score to distance
        return 1 - ari_score

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("ARI distance cannot be calculated from a pairwise distance matrix.")

class MutualInformationDistance(AbstractDistance):
    # Quantifies the amount of information obtained about one random variable through observing the other random variable, scaled to a fixed range between 0 and 1
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        X_mean, Y_mean = X.mean(axis=0), Y.mean(axis=0)
        return normalized_mutual_info_score(X_mean, Y_mean)

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("Mutual Information cannot be calculated from a pairwise distance matrix.")

class FowlkesMallowsDistance(AbstractDistance):
    # Based on the Fowlkes-Mallows index, a measure of clustering similarity
    def __init__(self, n_clusters=5) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        # Cluster the data
        kmeans_X = KMeans(n_clusters=self.n_clusters, random_state=42).fit(X_mean.reshape(-1, 1))
        kmeans_Y = KMeans(n_clusters=self.n_clusters, random_state=42).fit(Y_mean.reshape(-1, 1))
        # Get the cluster labels from both predictions and ground truth
        labels_X = kmeans_X.labels_
        labels_Y = kmeans_Y.labels_
        # Calculate the Fowlkes-Mallows Index
        fm_score = fowlkes_mallows_score(labels_X, labels_Y)
        return 1 - fm_score  # Convert to distance

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("Fowlkes-Mallows index cannot be calculated from a pairwise distance matrix.")

metric_dict = {
    "mse": pt.tools.Distance(metric="mse"),
    "mae": pt.tools.Distance(metric="mean_absolute_error"),
    "ks_test_distance": pt.tools.Distance(metric="ks_test"),
    "edistance": pt.tools.Distance(metric="edistance"),
    "cosine_distance": pt.tools.Distance(metric="cosine_distance"),
    "pearson_distance": pt.tools.Distance(metric="pearson_distance"),
    "euclidean_distance": pt.tools.Distance(metric="euclidean"),
    "classifier_proba": pt.tools.Distance(metric="classifier_proba"),
    "kendalltau_distance": pt.tools.Distance(metric="kendalltau_distance"),
    "spearman_distance": pt.tools.Distance(metric="spearman_distance"),
    "wasserstein": pt.tools.Distance(metric="wasserstein"),
    "sym_kldiv": pt.tools.Distance(metric="sym_kldiv"),
    "bhattacharyya_distance": BhattacharyyaDistance(),
    "jaccard_index": JaccardDistance(),
    "F1_score": F1ScoreDistance(),
    "adjusted_rand_index": ARIDistance(),
    "mutual_information": MutualInformationDistance(),
    "fowlkes_mallows_index": FowlkesMallowsDistance()
}
