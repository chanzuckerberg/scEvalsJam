import os
import scanpy as sc
import numpy as np
import torch
from perturbench import models
from importlib import reload
import perturbench

# ad = sc.read_h5ad("datasets/1gene-replogle-essential-split.h5ad")
ad = sc.read_h5ad("datasets/1gene-norman-split.h5ad")
# ad.obs['condition']  = ad.obs['perturbation_name'].apply(lambda x : x +"+ctrl" if x!='control' else 'ctrl')
# ad.var['gene_name'] = list(ad.var.index)
# ad.obs['cell_type'] = 'A549'
# ad = sc.read_h5ad('/home/wergillius/Project/GEARS/data/norman/perturb_processed.h5ad')

# train test split
model = models.gears_model.GearsModel(torch.device("cuda:0"), data_dir='datasets')
model.train(data=ad, 
            split_mode='single', 
            dataset_kws={'name':'norman-split'},  
            train_kws = {'epochs':2})

pert_genes = np.random.choice(model.model.pert_list, (10,)) 
pert_ad = model.predict(pert_genes)

# after training, reload the model 
model = models.gears_model.GearsModel(torch.device("cuda:0"), data_dir='datasets')
model.load_data(split_mode='single', dataset_kws={'name':'norman-split'})
model.load("norman-split_gears")