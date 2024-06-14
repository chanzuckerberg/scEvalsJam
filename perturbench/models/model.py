from abc import ABC, abstractmethod
import scipy as sp
import pathlib
import torch
from typing import List

from perturbench.dataset import PerturbationDataset


class PerturbationModel(ABC):
    """Class responsible for instantiating a model, training a model and performing a prediction."""

    @abstractmethod
    def __init__(self, device: torch.cuda.device, **kwargs) -> None:
        self.name = ''
        self.description = ''
        pass

    @abstractmethod
    def train(self, data: PerturbationDataset) -> None:
        pass

    @abstractmethod
    def predict(self, data: PerturbationDataset, perturbation: List[str]) -> sp.sparse.csr_matrix:
        """
        :param data:
            A PerturbationDataset where all cells are unperturbed (i.e. baseline), from which
            to make a prediction.
        :param perturbation:
            List of perturbations to predict where perturbations
            are encoded as described in PerturbationDataset.
        :return:
        """
        pass

    @abstractmethod
    def save(self) -> pathlib.Path:
        pass

    @abstractmethod
    def load(self, path: pathlib.Path) -> None:
        pass

    @abstractmethod
    def get_name(self) -> str:
        return self.name

    @abstractmethod
    def get_description(self) -> str:
        return self.description
