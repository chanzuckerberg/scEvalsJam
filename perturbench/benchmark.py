from typing import List

from models.model import PerturbationModel
from dataset import PerturbationDataset
from metrics import PerturbationMetric


class PerturbationBenchmark:
    """Responsible for comparing performance across different scenarios / models"""

    def __init__(self,
                 models: List[PerturbationModel] = [],
                 train_test_dict: dict = {},
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
        self.train_test_dict = train_test_dict
    
    def add_model(self, PerturbationModel):
        """Add a model to the list of perturbation benchmark"""

        self.models.append(PerturbationModel)

        return
    
    def add_dataset(self, train_test_dict):
        """Add a pair of train and test PerturbationDataset dataset 
        to the perturbation benchmark class"""

        ## TODO: Check this with data module
        # self.datasets.append(PerturbationDataset)
        self.train_test_dict = train_test_dict

        return

    def train(self):
        """Train each model in the list of perturbation benchmark"""

        for model in self.models:
            model.train(self.train_test_dict['train'])
            model.istrained = True
            print(f"Model {model.model_name} training completed successfully")

        return
    
    def predict(self, pertturbation_list = []):
        """Predict each model in the self.models list on self.train_test_dict['test'] data"""

        for model in self.models:
            model.predict(self.train_test_dict['test'],
                          pertturbation_list)
            print(f"Model {model.model_name} prediction completed successfully")

        return
    
    def calculate_metrics(self, adata_test, control_label = 'control', 
                          ground_truth_label = 'ground_truth',
                          pred_label = 'stimulated',
                          condition_label = 'condition',
                          deg_count = 100):
        """Calculate metrics for each model in the list of perturbation benchmark"""

        for model in self.models:
            
            # model.calculate_metrics()
            print(f"Model {model.model_name} metrics are calculated successfully")

        return
    
    def run(self):
        """Run the training, prediction and metric calculation for each
        model in the list of perturbation benchmark"""
        
        ## data processing and split to train/test TODO: implement this
        
        ## train all models
        self.train(self.train_test_dict['train'])
        
        ## predict all models
        self.predict(self.train_test_dict['test'], self.test_perturbation_list)
            
        ## calculate metrics
        self.calculate_metrics()
        return
    
    