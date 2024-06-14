import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse

"""
Authors:
Holly Whitfield
Ramon Vinas
"""



def get_de_genes(anndata_x: ad.AnnData, anndata_y: ad.AnnData, method: str, top_k: int, groupby_key='perturbation_name') -> dict:
    """
    Calculate differentially expressed genes between control and perturbation.

    ### Args:
    ----------
    anndata_x:
        AnnData object containing control cells. Data in .X should be sparse.csr_matrix. 
        Assumes both AnnData objects were normalised appropriately.
    anndata_y:
        AnnData object containing perturbed or predicted cells. Data in .X should be sparse.csr_matrix. 
        Assumes both AnnData objects were normalised appropriately.   
    method:
        String indicated method to pass to sc.tl.rank_genes_groups().
    top_k:
        Integer indicating how many top DE genes to return.
    
    """
    
    ## -- Check input
    if not sparse.issparse(anndata_x.X):
        raise Exception("Input anndata_x.X is not sparse.csr_matrix")
    if not sparse.issparse(anndata_y.X):
        raise Exception("Input anndata_y.X is not sparse.csr_matrix")
    
    ## -- Add .obs column indicating dataset
    anndata_x.obs["group"] = "control"
    anndata_y.obs["group"] = "perturbation"
    
    ## -- Concat AnnData objects
    try:
        ad_joint = ad.concat([anndata_x, anndata_y])
    except:
        raise Exception("Cannot concat AnnData objects")
    
    ## -- Calculate DE genes
    sc.tl.rank_genes_groups(ad_joint,
                            groupby=groupby_key,
                            reference='control',
                            method=method,
                            key_added=method,
                            rankby_abs=True,
                            n_genes=None,
                            use_raw=False
                            )
    de_df = sc.get.rank_genes_groups_df(ad_joint, group=None, key=method)
    
    ## -- Extract DE genes
    out = {}
    unique_perturbations = set(ad_joint.obs[groupby_key].unique()) - set(['control'])
    for pert in unique_perturbations:
        # Select DE results for perturbation pert
        subset_de_df = de_df[de_df['group'] == pert]

        # Extract DE genes
        up_genes = subset_de_df[subset_de_df["logfoldchanges"]>0]["names"][:top_k]
        dn_genes = subset_de_df[subset_de_df["logfoldchanges"]<0]["names"][:top_k]
        out[pert] = {"de_up":np.asarray(up_genes),"de_dn":np.asarray(dn_genes)}
    
    return out
    


### --- Example Usage:
## Get DE genes
#output_1 = get_de_genes(anndata_x, anndata_y, method = "wilcoxon", top_k = 100)
#output_2 = get_de_genes(anndata_x, anndata_z, method = "wilcoxon", top_k = 100)

## Calculate jaccard
# J_up = calculate_jaccard(output_1["de_up"], output_2["de_up"])
# J_dn = calculate_jaccard(output_1["de_dn"], output_2["de_dn"])
    
