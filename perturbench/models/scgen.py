import sys
sys.path.append('/Workspace/Users/lm25@sanger.ac.uk/scEvalsJam')
print(sys.path)
from perturbench.models import PerturbationModel
from perturbench.dataset import PerturbationDataset
import pathlib
import scipy as sp
import torch


class ScGenModel(PerturbationModel):
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
