import torch


def batch_log_mean(X: torch.Tensor, step_size: int = 1000):
    # Compute mean(log(X), dim=0) in batches (to be more memory efficient)
    log_sum = 0
    for i in range(0, X.shape[0], step_size):
        log_sum += torch.log(X[i : i + step_size] + 1).sum(0)
    log_mean = log_sum / X.shape[0]
    return log_mean


def batch_log_std(
    X: torch.Tensor, log_mean: torch.Tensor, correction: int = 1, step_size: int = 1000
):
    # Compute std(log(X), dim=0) in batches (to be more memory efficient)
    log_sum_sq_residual = 0
    for i in range(0, X.shape[0], step_size):
        log_sum_sq_residual += (
            (torch.log(X[i : i + step_size] + 1) - log_mean) ** 2
        ).sum(0)
    log_std = torch.sqrt(log_sum_sq_residual / (X.shape[0] - correction))
    return log_std
