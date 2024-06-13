from abc import ABC, abstractmethod
import scipy as sp
import pathlib
import torch

from perturbevals.dataset import PerturbationDataset


class PerturbationModel(ABC):
    """Class responsible for instantiating a model, training a model and performing a prediction."""

    @abstractmethod
    def __init__(self, device: torch.cuda.device, **kwargs) -> None:
        pass

    @abstractmethod
    def train(self, data: PerturbationDataset) -> None:
        pass

    @abstractmethod
    def predict(self) -> sp.sparse.csr_matrix:
        pass

    @abstractmethod
    def save(self) -> pathlib.Path:
        pass

    @abstractmethod
    def load(self, path: pathlib.Path) -> None:
        pass
