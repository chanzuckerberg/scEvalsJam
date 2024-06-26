import numpy as np
import scanpy as sc 
import anndata as ann 
import torch 

class MethodTransform:
    """Method-specific transforms for processed perturbation data"""
    
    def __init__(self, adata, gpu=False, config=None):
        self.adata = adata
        self.gpu = gpu
        self.config = config
        
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
            
    def process_data(self, method = None, celltype_key = None, perturbation_key = None):
        # Parse the config for key parameters 
        if self.config is not None:
            hvg = self.config['hvg']
            num_hvgs = self.config['num_hvgs']
        else:
            hvg = None
            num_hvgs = None
        
        # Current preprocessing assumes that the data is not normalized or log-transformed
        if method is None:
            raise Exception("Please specify a method.")
        if method is "scgen":
            self.adata.obs["batch_key"] = self.adata.obs[perturbation_key]
            self.adata.obs["labels_key"] = self.adata.obs[celltype_key]
            if hvg is True:
                sc.pp.highly_variable_genes(self.adata, n_top_genes = num_hvgs)
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
    
    def return_anndata(self):
        # Note that some methods will require more than just returning
        # the anndata object (e.g. scPreGAN)
        return self.adata
    
    