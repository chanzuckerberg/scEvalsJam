# Databricks notebook source
#create environment
!apt install -y python3.9-venv
!python -m venv ./venv/scgen_env
!source venv/scgen_env/bin/activate
!pip install anndata
!pip install Cython
!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# COMMAND ----------

import perturbench

# COMMAND ----------

!wget https://huggingface.co/datasets/scEvalsJam/datasets/resolve/7f7da698f89ba3dbfea4fcab12a286a6673f6871/1gene-norman-split.h5ad?download=true
!ls

# COMMAND ----------

sys.path

# COMMAND ----------

!pwd

# COMMAND ----------

sys.path

# COMMAND ----------

sys.path.append(os.path.abspath('..'))

# COMMAND ----------

import perturbench

# COMMAND ----------


