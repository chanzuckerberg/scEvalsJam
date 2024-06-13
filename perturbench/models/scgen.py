from perturbench import PerturbationModel
from perturbench import PerturbationDataset
import pathlib
import scipy as sp
import torch


class scGenModel(PerturbationModel):
    def __init__(self, device: torch.cuda.device, **kwargs) -> None:
        pass

    def train(self, data: PerturbationDataset) -> None:
        pass

    def predict(self) -> sp.sparse.csr_matrix:
        pass

    def save(self) -> pathlib.Path:
        pass

    def load(self, path: pathlib.Path) -> None:
        pass
