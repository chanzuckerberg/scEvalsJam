import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression


def get_mask_stats(mask_true: np.array, mask_estimated: np.array):
    """
    Estimate precision, recall, and F1 score between true and inferred masks
    under "best" reordering of estimated mask (permute to maximize true positives)

    Parameters
    ----------
    mask_true: boolean numpy array of true mask (from simulation)
    mask_estimated: boolean numpy array of estimated mask

    Returns
    -------
    Tuple of mask statistics and index for reordering estimated mask
    """
    mask_true = mask_true.astype(float)
    mask_estimated = mask_estimated.astype(float)
    tp = np.dot(mask_true.T, mask_estimated)
    idx = linear_sum_assignment(-1 * tp)
    stats = dict(
        precision=tp[idx].sum() / mask_estimated.sum(),
        recall=tp[idx].sum() / mask_true.sum(),
        f1=2 * tp[idx].sum() / (mask_estimated.sum() + mask_true.sum()),
    )
    estimated_mask_idx = idx[1]
    return stats, estimated_mask_idx


# All below copied from https://github.com/Genentech/sVAE/blob/main/svae/metrics.py
def get_linear_score(x, y):
    reg = LinearRegression().fit(x, y)
    return reg.score(x, y)


def linear_regression_metric(z, z_hat, num_samples=int(1e5), indices=None):

    score = get_linear_score(z_hat, z)
    # masking z_hat
    # TODO: this does not take into account case where z_block_size > 1
    if indices is not None:
        z_hat_m = z_hat[:, indices[-z.shape[0] :]]
        score_m = get_linear_score(z_hat_m, z)
    else:
        score_m = 0

    return score, score_m


def mean_corr_coef_np(x, y, method="pearson"):
    """
    A numpy implementation of the mean correlation coefficient metric.
    :param x: numpy.ndarray
    :param y: numpy.ndarray
    :param method: str, optional
            The method used to compute the correlation coefficients.
                The options are 'pearson' and 'spearman'
                'pearson':
                    use Pearson's correlation coefficient
                'spearman':
                    use Spearman's nonparametric rank correlation coefficient
    :return: float
    """
    d = x.shape[1]
    if method == "pearson":
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
    elif method == "spearman":
        cc = spearmanr(x, y)[0][:d, d:]
    else:
        raise ValueError("not a valid method: {}".format(method))
    cc = np.abs(cc)
    idx = linear_sum_assignment(-1 * cc)
    score = cc[idx]
    return score, score.mean(), idx
