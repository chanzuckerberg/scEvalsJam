from abc import ABC, abstractmethod
from perturbench.models import PerturbationDataset


class Scenario(ABC):
    """Responsible for creating a train/test split according to a given evaluation scenario."""

    @abstractmethod
    def __init__(self, data: PerturbationDataset):
        pass
