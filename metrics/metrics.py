import pertpy as pt
import numpy as np
from scipy import stats
from scipy.stats import entropy
from scipy.spatial.distance import cosine, pdist, squareform
from scipy.cluster.hierarchy import linkage, cophenet
from math import log, sqrt
from sklearn.metrics import adjusted_rand_score, f1_score, silhouette_score, mutual_info_score, fowlkes_mallows_score
from pertpy.tools._distances._distances import AbstractDistance


"""
Authors:
Shawn Fan
Artur Dox
Ang Li
Holly Whitfield
Ramon Vinas

Source: 
https://github.com/scverse/pertpy
https://pertpy.readthedocs.io/en/latest/_modules/pertpy/tools/_distances/_distances.html#Distance

"""

def jaccard_similarity(list1, list2):
    """
    Compute the Jaccard similarity between two lists.
    """
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2)) / len(set1.union(set2))


def mse(X: np.array, Y: np.array) -> float:
    """Mean squared distance between pseudobulk vectors (gene centroids).

    ### Args:
        - `X (np.array)`: ground truth
        - `Y (np.array)`: predictions
    """
    Distance = pt.tools.Distance(metric="mse")
    D = Distance(X, Y)
    return D


def mae(X: np.array, Y: np.array) -> float:
    """Absolute (Norm-1) distance between pseudobulk (gene centroids).

    ### Args:
        - `X (np.array)`: ground truth
        - `Y (np.array)`: predictions
    """
    Distance = pt.tools.Distance(metric="mean_absolute_error")
    D = Distance(X, Y)
    return D


def euclidean_distance(X: np.array, Y: np.array) -> float:
    """Euclidean distance between pseudobulk vectors (gene centroids).

    ### Args:
        - `X (np.array)`: ground truth
        - `Y (np.array)`: predictions
    """
    Distance = pt.tools.Distance(metric="euclidean")
    D = Distance(X, Y)
    return D


def ks_test_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Average of two-sided KS test statistic between two groups.

    ### Args:
        - `X (np.ndarray)`: ground truth
        - `Y (np.ndarray)`: predictions
    """
    Distance = pt.tools.Distance(metric="ks_test")
    D = Distance(X, Y)
    return D


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    In essence, it is twice the mean pairwise distance between cells of two groups minus
    the mean pairwise distance between cells within each group respectively. More information can
    be found in `Peidli et al. (2023) <https://doi.org/10.1101/2022.08.20.504663>`__.

    ### Args:
        - `X (np.ndarray)`: _description_
        - `Y (np.ndarray)`: _description_

    ### Returns:
        - `float`: _description_
    """
    Distance = pt.tools.Distance(metric="edistance")
    D = Distance(X, Y)
    return D


def cosine_distance(X: np.array, Y: np.array) -> float:
    """Cosine distance between pseudobulk vectors (gene centroids).

    ### Args:
        - `X (np.ndarray)`: control
        - `Y (np.ndarray)`: perturbations
    """
    Distance = pt.tools.Distance(metric="cosine_distance")
    D = Distance(X, Y)
    return D


def cosine_similarity(X: np.array, Y: np.array) -> float:
    """Cosine similarity between pseudobulk vectors (gene centroids).

    ### Args:
        - `X (np.ndarray)`: control
        - `Y (np.ndarray)`: perturbations
    """
    return cosine(X.mean(axis=0), Y.mean(axis=0))


def pearson_distance(X: np.array, Y: np.array) -> float:
    """Pearson distance between the means of cells from two groups.

    ### Args:
        - `X (np.ndarray)`: control
        - `Y (np.ndarray)`: perturbations
    """
    Distance = pt.tools.Distance(metric="pearson_distance")
    D = Distance(X, Y)
    return D


def pearson_correlation(X: np.array, Y: np.array) -> float:
    """Pearson distance between the means of cells from two groups.

    ### Args:
        - `X (np.ndarray)`: control
        - `Y (np.ndarray)`: perturbations
    """
    return stats.pearsonr(X.mean(axis=0), Y.mean(axis=0))[0]


def r2_distance(X: np.array, Y: np.array) -> float:
    """Coefficient of determination across genes between pseudobulk vectors (gene centroids).

    ### Args:
        - `X (np.ndarray)`: control
        - `Y (np.ndarray)`: perturbations
    """
    Distance = pt.tools.Distance(metric="r2_distance")
    D = Distance(X, Y)
    return D


def classifier_control_proba(X: np.array, Y: np.array) -> float:
    """Average of the classification probability of the perturbation for a binary classifier.

    ### Args:
        - `X (np.ndarray)`: control
        - `Y (np.ndarray)`: perturbations
    """
    Distance = pt.tools.Distance(metric="classifier_proba")
    D = Distance(X, Y)
    return D


class BhattacharyyaDistance(AbstractDistance):
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape")
        BC = np.sum(np.sqrt(X * Y))
        return -log(BC)

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("Bhattacharyya distance cannot be calculated from precomputed matrix directly.")

class JaccardDistance(AbstractDistance):
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = True

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        intersection = np.sum(np.logical_and(X, Y))
        union = np.sum(np.logical_or(X, Y))
        return 1 - intersection / union
    
    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        intersection = P[idx, :][:, idx].sum()
        union = P.sum() - P[~idx, :][:, ~idx].sum()
        return 1 - intersection / union
    
class F1ScoreDistance(AbstractDistance):
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same length")
        return f1_score(X, Y)
    
    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("F1 score cannot be calculated from a pairwise distance matrix.")

class ARIDistance(AbstractDistance):
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        if X.ndim != 1 or Y.ndim != 1:
            raise ValueError("Both X and Y must be 1-dimensional arrays representing cluster labels.")
        ari_score = adjusted_rand_score(X, Y)
        # ARI ranges from -1 to 1, where 1 means perfect agreement.
        # To use it as a "distance", we can take 1 minus the score.
        # Hence, lower values indicate higher distance (lower similarity).
        return 1 - ari_score

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("ARI cannot be calculated from a pairwise distance matrix.")

class SilhouetteDistance(AbstractDistance):
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        # Silhouette score ranges from -1 to 1, where 1 means better defined clusters
        score = silhouette_score(X, labels)
        return 1 - score  # Convert to a distance measure

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("Silhouette score cannot be calculated from a pairwise distance matrix.")

class MutualInformationDistance(AbstractDistance):
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        mi = mutual_info_score(X.ravel(), Y.ravel())
        return -mi  # Use the negative because higher MI indicates more similarity

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("Mutual Information cannot be calculated from a pairwise distance matrix.")

class CopheneticDistance(AbstractDistance):
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, **kwargs) -> float:
        Z = linkage(X, 'ward')
        cophenet_coef, _ = cophenet(Z, pdist(X))
        return 1 - cophenet_coef  # Convert to distance

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("Cophenetic correlation cannot be calculated from a pairwise distance matrix.")

class GiniCoefficient(AbstractDistance):
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, **kwargs) -> float:
        sorted_X = np.sort(X.flatten())
        n = len(sorted_X)
        cum_X = np.cumsum(sorted_X, dtype=float)
        index = np.arange(1, n+1)
        gini = (n + 1 - 2 * np.sum(cum_X / cum_X[-1] * index / n))
        return gini

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("Gini coefficient cannot be calculated from a pairwise distance matrix.")

class EntropyDistance(AbstractDistance):
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, **kwargs) -> float:
        # Calculate entropy and return as distance (since higher entropy indicates more disorder)
        return entropy(X.flatten())

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("Entropy cannot be calculated from a pairwise distance matrix.")

class FowlkesMallowsDistance(AbstractDistance):
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        fm_score = fowlkes_mallows_score(X, Y)
        return 1 - fm_score  # Convert to distance

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("Fowlkes-Mallows index cannot be calculated from a pairwise distance matrix.")