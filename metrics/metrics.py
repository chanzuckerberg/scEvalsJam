# Implementation of other metrics as extensions to pertpy package
# Created by: Ang Li
# Date: June 13, 2024

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import adjusted_rand_score, f1_score, mutual_info_score, fowlkes_mallows_score
from sklearn.cluster import KMeans
from pertpy.tools._distances._distances import AbstractDistance

metric_dict = {
    "mse": pt.tools.Distance(metric="mse"),
    "mae": pt.tools.Distance(metric="mean_absolute_error"),
    "ks_test_distance": pt.tools.Distance(metric="ks_test"),
    "edistance": pt.tools.Distance(metric="edistance"),
    "cosine_distance": pt.tools.Distance(metric="cosine_distance"),
    "pearson_distance": pt.tools.Distance(metric="pearson_distance"),
    "euclidean_distance": pt.tools.Distance(metric="euclidean_distance"),
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

class BhattacharyyaDistance(AbstractDistance):
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
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        # Clustering the data
        kmeans_X = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans_Y = KMeans(n_clusters=self.n_clusters, random_state=42)
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
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        x, y = X.mean(axis=0), Y.mean(axis=0)
        return mutual_info_score(x, y)

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("Mutual Information cannot be calculated from a pairwise distance matrix.")

class FowlkesMallowsDistance(AbstractDistance):
    def __init__(self, n_clusters=5) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        # Cluster the data
        kmeans_X = KMeans(n_clusters=self.n_clusters, random_state=42).fit(X)
        kmeans_Y = KMeans(n_clusters=self.n_clusters, random_state=42).fit(Y)
        # Get the cluster labels from both predictions and ground truth
        labels_X = kmeans_X.labels_
        labels_Y = kmeans_Y.labels_
        # Calculate the Fowlkes-Mallows Index
        fm_score = fowlkes_mallows_score(labels_X, labels_Y)
        return 1 - fm_score  # Convert to distance

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("Fowlkes-Mallows index cannot be calculated from a pairwise distance matrix.")
