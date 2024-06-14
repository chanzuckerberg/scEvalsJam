from abc import ABC, abstractmethod
from typing import Tuple

from perturbench.dataset import PerturbationDataset


class PerturbationScenario:

    @abstractmethod
    def __init__(self) -> None:
        self.name = "undefined"
        self.description = "undefined"
        pass

    @abstractmethod
    def split(self, data: PerturbationDataset) -> Tuple[PerturbationDataset, PerturbationDataset]:
        pass

    @abstractmethod
    def get_name(self) -> str:
        return self.name

    @abstractmethod
    def get_description(self) -> str:
        return self.description
