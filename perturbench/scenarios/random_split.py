from typing import Tuple
import numpy as np
import random

from perturbench.dataset import PerturbationDataset
from perturbench.scenarios import PerturbationScenario


class RandomSplitScenario(PerturbationScenario):

    def __init__(self) -> None:
        self.name = 'Random splitting'
        self.description = 'Randomely splits the input data'
        pass

    def split(self, dataset: PerturbationDataset) -> Tuple[PerturbationDataset, PerturbationDataset]:
        anndata = dataset.anndata()
        test_index = random.sample(range(anndata.shape[0]),
                                   int(anndata.shape[0] / 5))
        train_index = \
            [x for x in range(anndata.shape[0])
             if x not in test_index]
        train_adata = anndata[train_index]
        test_adata = anndata[test_index]

        train_dataset = PerturbationDataset(train_adata, 'perturbation', dataset.covariates.columns.tolist())
        test_dataset = PerturbationDataset(test_adata, 'perturbation', dataset.covariates.columns.tolist())

        if any([x is None for x in test_dataset.get_perturbations()]):
            raise ValueError("Test dataset should only contain control cells")

        return train_dataset, test_dataset

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

