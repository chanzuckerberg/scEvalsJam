# Databricks notebook source
#create environment
!apt install -y python3.9-venv
!python -m venv ./venv/scgen_env
!source venv/scgen_env/bin/activate
!pip install anndata
!pip install Cython
!pip install scikit-learn
!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install git+https://github.com/theislab/scgen.git

# COMMAND ----------

!pip install scvi-tools

# COMMAND ----------

# Install perturbench
import os
os.chdir('/Workspace/Users/lm25@sanger.ac.uk/scEvalsJam/perturbench')
!pip install -e .

# COMMAND ----------

!source venv/scgen_env/bin/activate

# COMMAND ----------

import os
import scgen
import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split

%reload_ext autoreload
%autoreload 2
from perturbench.models import scgen as scgen_perturbench


# COMMAND ----------

from perturbench.models import scgen as scgen_perturbench

# COMMAND ----------

# Load and prepare anndata
adata_dict = {}
adata = ad.read_h5ad('/dbfs/scEvalsJam/1gene-norman-split.h5ad')
# Log-normalised counts (were log-tranformed before)
adata.layers['counts'] = np.expm1(adata.X)
adata.X = adata.layers['counts'].copy()
sc.pp.normalize_total(adata, target_sum = 10000)
sc.pp.log1p(adata)

# Create cell type column
adata.obs['cell_type'] = 'cell_line'

# COMMAND ----------

#create a dictionnary
adata_train_dict = {}
adata_test_dict = {}
for perturbation in adata.obs['perturbation_name'].unique():
    if perturbation != 'control':
        adata_filtered = adata[adata.obs['perturbation_name'].isin(['control',perturbation])].copy()

        # Split into train and test set
        train_idx, test_idx = train_test_split(adata_filtered.obs_names, train_size = 0.8, test_size = 0.2, stratify = adata_filtered.obs['perturbation_name'].tolist())
        adata_filtered.obs['train_or_test'] = ['train' if n in train_idx.tolist() else 'test' for n in adata_filtered.obs_names]

        adata_train = adata_filtered[~((adata_filtered.obs['train_or_test'] == 'test') & (adata_filtered.obs['perturbation_name'] == perturbation))].copy()
        adata_truth = adata_filtered[~adata_filtered.obs_names.isin(adata_train.obs_names)].copy()
        adata_train_dict[perturbation] = adata_train
        adata_test_dict[perturbation] = adata_truth

# COMMAND ----------





# COMMAND ----------

result_pred = []
result_delta = []
for perturbation, adata_train in adata_train_dict.items():
    model_params = {
        'train_set':adata_train,
        'batch_key':'perturbation_name',
        'labels_key':'train_or_test',
        'ctrl_key':'control',
        'stim_key':perturbation,
        'cond_to_predict':'test',
        'device':'cuda',
        'save_model_path': '../saved_models/model_perturbation_prediction.pt',
    }
    model = scgen_perturbench.ScGenModel(**model_params)
    model.train()
    pred,delta = model.predict()
    result_pred.append(pred)
    result_delta.append(delta)

# COMMAND ----------


