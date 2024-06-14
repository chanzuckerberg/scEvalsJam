from models.model import PerturbationModel
from dataset import PerturbationDataset
from typing import List


class PerturbationBenchmark:
    """Responsible for comparing performance across different scenarios / models"""

    def __init__(self,
                 models: List[PerturbationModel] = [],
                 datasets: List[PerturbationDataset] = [],
                 metric: List[str] = ['r2', 'mse', 'mae'],
                 gene_subset: List[str] = ['all_genes'],
                 **kwargs):
        """
        :param models:
            List of models to compare.
        :param datasets:
            List of datasets to compare.
        :param metric:
            List of metrics to compare.
        :param gene_subset:
            List of gene subsets to compare.
        """
        self.models = models
        self.datasets = datasets
        pass
    
    def add_model(self, PerturbationModel):
        """Add a model to the list of perturbation benchmark"""

        self.models.append(PerturbationModel)

        pass
    
    def add_data(self, PerturbationDataset):
        """Add a dataset to the list of perturbation benchmark"""
        ## Check this with data module
        
        pass
    
    def train(self, train_data: PerturbationDataset):
        """Train each model in the list of perturbation benchmark"""

        for model in self.models:
            model.train(train_data)
            model.istrained = True
            print(f"Model {model.model_name} is trained successfully")

        pass
    
    def predict(self, test_data: PerturbationDataset, perturbation: List[str]):
        """Predict each model in the list of perturbation benchmark"""

        for model in self.models:
            model.predict(test_data, perturbation)
            print(f"Model {model.model_name} is predicted successfully")

        pass
    
    def calculate_metrics(self):
        """Calculate metrics for each model in the list of perturbation benchmark"""

        for model in self.models:
            model.calculate_metrics()
            print(f"Model {model.model_name} metrics are calculated successfully")

        pass
    
    def run(self):
        """Run the training, prediction and metric calculation for each
        model in the list of perturbation benchmark"""
        
        ## data processing and split to train/test TODO: implement this
        
        ## train all models
        self.train(train_data)
        
        ## predict all models
        self.predict(test_data, perturbation)
            
        ## calculate metrics
        self.calculate_metrics()
        pass
    
    