from typing import List
from tqdm import tqdm
import scipy as sp

from models.model import PerturbationModel
from dataset import PerturbationDataset
# from metrics import PerturbationMetric
from metrics.de import *
from metrics.compute_metrics import calc_metrics, metric_dict



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
    
    def calculate_metrics(self, adata_test,
                          control_label = 'control', 
                          target_pert_list = [],
                          condition_label = 'condition',
                          deg_count = 100):
        """Calculate metrics for each model in the list of perturbation benchmark
        
        :param adata_test:
            Anndata object of the test dataset, assumed to contain test cell groups
        :param control_label:
            Label of the control condition
        :param target_pert_list:
            List of target perturbations to compare
        :param condition_label:
            Label of the condition column in the adata object
        :param deg_count:
            Number of DEGs to consider
        
        """
        print("type of adata_test.X", type(adata_test.X))
        print(type(adata_test.X.toarray()))
        adata_test.X = sp.sparse.csr_matrix(adata_test.X.toarray())
        
        if target_pert_list == []:
            target_pert_list = adata_test.obs[condition_label].unique()
            # remove control_label from the list
            target_pert_list = [x for x in target_pert_list if x != control_label]
        
        adata_control = adata_test[adata_test.obs[condition_label] == control_label]
        adata_ground_truth = adata_test[adata_test.obs[condition_label].isin(target_pert_list)]
        
        # Run DE analysis
        de_genes_gt = get_de_genes(
            adata_control,
            adata_ground_truth,
            method = "wilcoxon",
            top_k = deg_count,
            groupby_key = condition_label)

        for model in self.models:

            de_genes_pred = get_de_genes(
                adata_control,
                model.adata_pred,
                method = "wilcoxon",
                top_k = deg_count,
                groupby_key = condition_label)
            
            ### --- Calculate metrics per scenario
            unique_perturbations = adata_test.obs[condition_label].unique()

            ## Calculate metrics
            results_list = [calc_metrics(
                adata_ground_truth, model.adata_pred, adata_control,
                pert, metric_dict, de_genes_gt, de_genes_pred,
                de_subset = None) for pert in tqdm(unique_perturbations)]
            ## Build dataframe
            results_df = pd.DataFrame(
                results_list,
                index=[i for i in unique_perturbations],
                columns=list(metric_dict.keys())+["Jaccard_de_up", "Jaccard_de_dn"])

            ## Remove redundant metrics
            redundant_column_mask = results_df.columns.str.contains('ec') & results_df.columns.str.contains('mse|mae|euclidean_distance|bhattacharyya_distance')
            results_df = results_df.loc[:, ~redundant_column_mask]
            
            # model.calculate_metrics()
            print(results_df)
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
    
    