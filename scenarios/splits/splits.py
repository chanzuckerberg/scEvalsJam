import numpy as np 
import pandas as pd 

class Scenarios:
    """Generic class for defining different scenarios for splitting data"""
    
    def __init__(self):
        pass
        
    def create_scenario(self, scenario):
        self.scenario_instance = scenario(self.adata, self.scenario_config)
        
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
            config: dictionary with keys:
                method: str, method that data was preprocessed with 
                test_size: float [0-1], proportion of data to be used for testing
                val_size: float [0-1], proportion of data to be used for validation
                scenario_name: str, name of the scenario
                train_dataset: str, name of the training dataset
                test_dataset: str, name of the testing dataset
                seed: int, random seed for reproducibility
    """
    
    def __init__(self, adata, config):
        self.adata = adata
        self.config = config
        self.parse_config()
        
    def parse_config(self):
        self.method = self.config['method']
        self.test_size = self.config['test_size']
        self.val_size = self.config['val_size']
        self.scenario_name = self.config['scenario_name']
        self.train_dataset = self.config['train_dataset']
        self.test_dataset = self.config['test_dataset']
        self.seed = self.config['seed']
    
    def 
        
    def split_data(self):
        
        
    
        
    
    
    

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