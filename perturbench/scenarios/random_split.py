from typing import Tuple
import numpy as np
import random

from perturbench.dataset import PerturbationDataset
from perturbench.scenarios import PerturbationScenario


class RandomSplitScenario(PerturbationScenario):

    def __init__(self) -> None:
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
        return train_dataset, test_dataset
