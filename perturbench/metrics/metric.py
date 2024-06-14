from abc import ABC, abstractmethod

from perturbench.dataset import PerturbationDataset


class PerturbationMetric(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def compute(self,
                control: PerturbationDataset,
                ground_truth: PerturbationDataset,
                prediction: PerturbationDataset
                ) -> int:
        pass
