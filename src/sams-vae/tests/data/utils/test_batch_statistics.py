import torch
from torch.distributions import Poisson

from sams_vae.data.utils.batch_statistics import batch_log_mean, batch_log_std


class TestBatchStatistics:
    def test_batch_log_statistics(self):
        torch.manual_seed(0)
        x = Poisson(15).sample((100, 15))
        logx = torch.log(x + 1)

        log_mean = batch_log_mean(x, step_size=10)
        assert torch.all(torch.isclose(log_mean, logx.mean(0)))
        log_std = batch_log_std(x, log_mean, step_size=10)
        assert torch.all(torch.isclose(log_std, logx.std(0)))
