
from perturbench.metrics import PerturbationMetric
from perturbench.dataset import PerturbationDataset
import numpy as np


class AverageDifferenceMetric(PerturbationMetric):

    def __init__(self):
        self.name = 'Average of difference metric'
        self.description = 'Average of difference metric'
        pass

    def compute(self,
                control: PerturbationDataset,
                ground_truth: PerturbationDataset,
                prediction: PerturbationDataset
                ) -> int:

        gene_wise_average_truth = ground_truth.raw_counts
        gene_wise_average_prediction = prediction.raw_counts
        gene_wise_difference = gene_wise_average_truth - gene_wise_average_prediction
        result = np.mean(gene_wise_difference)
        return result

    def get_name(self) -> str:
        return 'Average of difference metric'

    def get_description(self) -> str:
        return 'Average of difference metric'

