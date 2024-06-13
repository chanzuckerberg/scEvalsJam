import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

class Scenarios:
    """Generic class for defining different scenarios for splitting data"""
    
    def __init__(self, scenario, adata, scenario_config):
        self.scenario_instance = scenario(adata, scenario_config)
        
    def return_data(self):
        self.scenario_instance.format_scenario()
        outs = self.scenario_instance.split_data()
        return outs

class Scenario1(Scenarios):
    """
    Scenario 1: Forward perturbation prediction, same cell-type/cell-line, different datasets. 
        All of the perturbations in the test set are seen in the training set in a different
        dataset but within the same tissue/cell-line context.
        
        Parameters:
            adata_train: AnnData object, training dataset - validation will be performed on this dataset
            adata_test: AnnData object, testing dataset - testing will be performed on this dataset
            config: dictionary with keys:
                method: str, method that data was preprocessed with 
                test_size: float [0-1], proportion of data to be used for testing
                val_size: float [0-1], proportion of data to be used for validation
                scenario_name: str, name of the scenario
                train_dataset: str, name of the training dataset
                test_dataset: str, name of the testing dataset
                perturbation_key: str, key in the obs dataframe that contains the perturbation 
                    names
                cell_type_key: str, key in the obs dataframe that contains the cell type names
                min_cells: int, minimum number of cells for a gene in train and test
                    for it to be included in the analysis
                seed: int, random seed for reproducibility
    """
    
    def __init__(self, adata_train, adata_test, config):
        self.adata_train = adata_train
        self.adata_test = adata_test
        self.config = config
        self.parse_config()
        
    def parse_config(self):
        # Load the config parameters
        self.method = self.config['method']
        self.test_size = self.config['test_size']
        self.val_size = self.config['val_size']
        self.scenario_name = self.config['scenario_name']
        self.train_dataset = self.config['train_dataset']
        self.test_dataset = self.config['test_dataset']
        self.perturbation_key = self.config['perturbation_key']
        self.cell_type_key = self.config['cell_type_key']
        self.min_cells = self.config['min_cells']
        self.seed = self.config['seed']
        
    def format_scenario(self):
        # Ensure that the number and names of genes are equivalent between
        # test and train
        assert np.all(self.adata_train.var_names == self.adata_test.var_names), \
            "Genes in training and testing datasets are not the same"
        
        # Ensure that the perturbation names are the same between test and train
        assert np.all(self.adata_train.obs[self.perturbation_key].unique() == \
                        self.adata_test.obs[self.perturbation_key].unique()), \
            "Perturbations in training and testing datasets are not the same"
        
        # Add train and test quantifiers to the anndata objects
        self.adata_train.obs['split'] = 'train'
        self.adata_test.obs['split'] = 'test'
        
        # Add dataset names to the anndata objects if they don't already exist
        if 'dataset' not in self.adata_train.obs.columns:
            self.adata_train.obs['dataset'] = self.train_dataset
        if 'dataset' not in self.adata_test.obs.columns:
            self.adata_test.obs['dataset'] = self.test_dataset
            
    def split_test_train(self, adata, test_size, stratification_key):
        # Extract the anndata obs dataframe with indices and convert to array
        adata_arr = adata.obs.values
        adata_indices = adata.obs.index.values
        stratification_arr = adata.obs[stratification_key].values
        
        # Perform the train-test split
        adata_arr_train, adata_arr_test = train_test_split(
            adata_arr, test_size = test_size, train_size =  1-test_size, 
            stratify = stratification_arr, shuffle=True, random_state=self.seed
        )
        
        # [Add test train split here]
        
        
    def split_data(self):
        # Initialize seed for reproducibility
        np.random.seed(self.seed)
        
        # Calculate perturbation value counts for train and test
        train_counts = self.adata_train.obs[self.perturbation_key].value_counts()
        test_counts = self.adata_test.obs[self.perturbation_key].value_counts()
        
        # Filter the train and test data based on the minimum number of cells
        # for each perturbation
        train_perturbations = train_counts[train_counts >= self.min_cells].index
        test_perturbations = test_counts[test_counts >= self.min_cells].index
        self.adata_train = self.adata_train[
            self.adata_train.obs[self.perturbation_key].isin(train_perturbations), :
        ]
        self.adata_test = self.adata_test[
            self.adata_test.obs[self.perturbation_key].isin(test_perturbations), :
        ]
        
        # Split the train data into validation and training sets
        adata_train_frame = self.adata_train.obs
        adata_test_frame = self.adata_test.obs
        
        
    
        
    
    
    

class ScenarioSplits:
    """Class for scenario-specific test-train splits"""
    
    def __init__(self, adata):
        self.adata = adata
        
    def scenario_1(self, test_size, remove_genes):
        

    def split_data(self, scenario = 1, test_size = 0.1, remove_genes = None):
        if scenario == 1:
            return self.scenario_1(test_size, remove_genes)
        else:
            pass