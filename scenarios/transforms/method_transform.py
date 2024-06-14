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
            
    def process_data(self, method = None, sample_key = None, perturbation_key = None, gene_name_key = None, cell_type_key = None):
        # Current preprocessing assumes that the data is not normalized or log-transformed
        if method is None:
            raise Exception("Please specify a method.")
        if method is "scgen":
            self.adata.obs["batch_key"] = self.adata.obs[sample_key]
            self.adata.obs["labels_key"] = self.adata.obs[perturbation_key]
        elif method is "cpa":
            pass
        elif method is "chemcpa":
            pass
        elif method is "scpregan":
            pass
        elif method is "gears":
            # https://github.com/snap-stanford/GEARS/blob/master/demo/data_tutorial.ipynb
            self.adata.var['gene_name'] = gene_name_key
            # TODO: reformat condition key
            self.adata.obs['condition'] = perturbation_key
            self.adata.obs['cell_type'] = cell_type_key
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
            self.adata.X = torch.Tensor(self.adata.X).cuda()
        else:
            self.adata.X = torch.Tensor(self.adata.X)
    
    def return_anndata(self):
        return self.adata
    
    