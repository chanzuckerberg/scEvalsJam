import numpy as np
import scanpy as sc 
import anndata as ann 
import torch 

class MethodTransform:
    """Method-specific transforms for processed perturbation data"""
    
    def __init__(self, adata, gpu=False):
        self.adata = adata
        self.gpu = gpu
        
        # Check anndata object 
        if not isinstance(adata, ann.AnnData):
            raise Exception("Please input an AnnData object.")
        # Check if gpu is available
        if gpu is True:
            if torch.cuda.is_available():
                self.gpu = True
            else:
                raise Exception("GPU not available. Please set gpu = False.")
        else:
            self.gpu = False
            
    def process_data(self, method = None):
        if method is None:
            raise Exception("Please specify a method.")
        if method is "scgen":
            pass
        elif method is "cpa":
            pass
        elif method is "chemcpa":
            pass
        elif method is "scpregan":
            pass
        elif method is "gears":
            pass
        elif method is "cellot":
            pass
        elif method is "graphvci":
            pass
        elif method is "biolord":
            pass
        elif method is "samsvae":
            pass
        elif method is "scgpt":
            pass
        elif method is "cellplm":
            pass
        else:
            raise Exception("Method not found.")
        
    def anndata_to_tensor(self):
        if self.gpu is True:
            self.adata.X = torch.tensor(self.adata.X).cuda()
        else:
            self.adata.X = torch.tensor(self.adata.X)
    
    def return_anndata(self):
        return self.adata
    
    