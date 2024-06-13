import pertpy as pt
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine

"""
Authors:
Shawn Fan
Artur Dox

Source: 
https://github.com/scverse/pertpy
https://pertpy.readthedocs.io/en/latest/_modules/pertpy/tools/_distances/_distances.html#Distance

"""


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
