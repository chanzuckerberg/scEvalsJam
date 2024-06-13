from abc import ABC, abstractmethod
import scipy as sp
import pathlib
import torch
from typing import List

from perturbench import PerturbationModel
from perturbench import PerturbationDataset


class Scenario(ABC):
    """Responsible for creating a train/test split according to a given evaluation scenario."""

    def __init__(self, data: PerturbationDataset):
        pass
