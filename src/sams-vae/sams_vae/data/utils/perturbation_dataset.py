from typing import Iterable, Optional, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset


class PerturbationDataSample(TypedDict):
    idx: int
    X: torch.Tensor
    D: torch.Tensor


class SCRNASeqPerturbationDataSample(PerturbationDataSample):
    library_size: int


class PerturbationDataset(Dataset):
    """Base class for perturbation pytorch datasets."""

    def __getitem__(self, idx: int) -> PerturbationDataSample:
        raise NotImplementedError

    def get_dosage_obs_per_dim(self) -> torch.Tensor:
        """
        Return tensor with shape (n_perturbation,) that contains the
        number of observed samples with non-zero dosages for each perturbation
        Used for computing ELBO with plates
        """
        raise NotImplementedError

    def convert_idx_to_ids(self, idx: np.array) -> np.array:
        """
        Convert array of sample index values to sample IDs
        """
        raise NotImplementedError


class TensorPerturbationDataset(PerturbationDataset):
    """PerturbationDataset for X and D stored in memory as pytorch tensors"""

    def __init__(
        self,
        X: torch.Tensor,
        D: torch.Tensor,
        ids: Optional[Iterable] = None,
    ):
        self.X = X  # observations
        self.D = D  # perturbations
        if ids is None:
            self.ids = np.arange(len(X))
        else:
            self.ids = np.array(ids)

        self.D_obs_per_dim = (self.D != 0).sum(0)

        self.library_size = self.X.sum(1)

    def __getitem__(self, idx: int) -> PerturbationDataSample:
        return dict(idx=idx, X=self.X[idx], D=self.D[idx])

    def __len__(self):
        return len(self.X)

    def get_dosage_obs_per_dim(self):
        return self.D_obs_per_dim

    def convert_idx_to_ids(self, idx: np.array) -> np.array:
        return self.ids[idx]


class SCRNASeqTensorPerturbationDataset(TensorPerturbationDataset):
    def __getitem__(self, idx: int) -> SCRNASeqPerturbationDataSample:
        return dict(
            idx=idx, X=self.X[idx], D=self.D[idx], library_size=self.library_size[idx]
        )
