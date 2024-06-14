import numpy as np
import scipy as sp
import pathlib
import torch
from typing import List
import uuid
import os

from perturbench.models import PerturbationModel
from perturbench.dataset import PerturbationDataset


class RandomModel(PerturbationModel):
    """Samples from a normal distribution"""
    def __init__(self, device: torch.cuda.device, **kwargs) -> None:
        self.kwargs = kwargs
        self.device = device
        self.name = "Random model"
        self.description = "Generates a random prediction that is normally distributed"
        pass

    def train(self, data: PerturbationDataset) -> None:
        self.model = "trained!"

    def predict(self, data: PerturbationDataset, perturbation: List[str]) -> sp.sparse.csr_matrix:
        raw_data = data.raw_counts()
        shape = raw_data.shape
        predicted_values = np.random.normal(size=shape)
        sparse_predicted_values = sp.sparse.csr_matrix(predicted_values)
        return sparse_predicted_values

    def save(self) -> pathlib.Path:
        path = pathlib.Path(os.path.join("artefacts", "models", str(uuid.uuid4())))
        with open(os.path.join(path, "model.txt"), "w") as file:
            file.write(self.model)
        return path

    def load(self, path: pathlib.Path) -> None:
        with open(os.path.join(path, "model.txt"), "r") as file:
            self.model = file.read()

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description