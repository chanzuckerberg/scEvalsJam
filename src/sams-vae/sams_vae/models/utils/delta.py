import torch
from pyro.distributions import Delta


class DeltaDist(Delta):
    @property
    def mode(self) -> torch.Tensor:
        return self.mean
