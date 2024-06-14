from abc import ABC, abstractmethod
from typing import Tuple

from perturbench.dataset import PerturbationDataset


class PerturbationScenario:

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def split(self, data: PerturbationDataset) -> Tuple[PerturbationDataset, PerturbationDataset]:
        pass
