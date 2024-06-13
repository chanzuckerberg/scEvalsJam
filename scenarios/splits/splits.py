import numpy as np 
import pandas as pd 

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