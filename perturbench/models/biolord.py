from perturbench import PerturbationModel
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import warnings
import os
import biolord
from ..utils import (
    make_GO, get_map, bool2idx, repeat_n
)
import anndata as ad


np.random.seed(42)

# -----------------------------------------------------------------------
# Configuration
# from https://github.com/nitzanlab/biolord_reproducibility/blob'
# '/main/scripts/biolord/norman/norman_optimal_config.py'
# -----------------------------------------------------------------------


varying_arg = {
    "seed": 42,
    "unknown_attribute_noise_param": 0.2,
    "use_batch_norm": False,
    "use_layer_norm": False,
    "step_size_lr": 45,
    "attribute_dropout_rate": 0.0,
    "cosine_scheduler":True,
    "scheduler_final_lr":1e-5,
    "n_latent":32,
    "n_latent_attribute_ordered": 32,
    "reconstruction_penalty": 10000.0,
    "attribute_nn_width": 64,
    "attribute_nn_depth" :2,
    "attribute_nn_lr": 0.001,
    "attribute_nn_wd": 4e-8,
    "latent_lr": 0.01,
    "latent_wd": 0.00001,
    "decoder_width": 32,
    "decoder_depth": 2,
    "decoder_activation": True,
    "attribute_nn_activation": True,
    "unknown_attributes": False,
    "decoder_lr": 0.01,
    "decoder_wd": 0.01,
    "max_epochs":200,
    "early_stopping_patience": 200,
    "ordered_attributes_key": "perturbation_neighbors",
    "n_latent_attribute_categorical": 16,
}

# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

class BioLord(PerturbationModel):
    def __init__(self, 
                 adata : ad.AnnData, 
                 config : dict = varying_arg,
                 ):
        self.setup_config(config)
        self.validate_config()

        self.adata = self.preprocess(adata)

        self.model = biolord.Biolord(
            adata=self.adata,
            n_latent=self.config['n_latent'],
            model_name=self.config['model_name'],
            module_params=self.module_params,
            train_classifiers=False,
            split_key=self.split_key,
        )

    def setup_config(self, varying_arg):
        self.config = varying_arg
        self.split_key = f'split_{self.config["split_id"]}'

        self.module_params = {
            "attribute_nn_width":  varying_arg["attribute_nn_width"],
            "attribute_nn_depth": varying_arg["attribute_nn_depth"],
            "use_batch_norm": varying_arg["use_batch_norm"],
            "use_layer_norm": varying_arg["use_layer_norm"],
            "attribute_dropout_rate":  varying_arg["attribute_dropout_rate"],
            "unknown_attribute_noise_param": varying_arg["unknown_attribute_noise_param"],
            "seed": varying_arg["seed"],
            "n_latent_attribute_ordered": varying_arg["n_latent_attribute_ordered"],
            "n_latent_attribute_categorical": varying_arg["n_latent_attribute_categorical"],
            "reconstruction_penalty": varying_arg["reconstruction_penalty"],
            #"unknown_attribute_penalty": varying_arg["unknown_attribute_penalty"],
            # not defined in python script with optimal config for norman 
            "decoder_width": varying_arg["decoder_width"],
            "decoder_depth": varying_arg["decoder_depth"],
            "decoder_activation": varying_arg["decoder_activation"],
            "attribute_nn_activation": varying_arg["attribute_nn_activation"],
            "unknown_attributes": varying_arg["unknown_attributes"],
        }

        self.trainer_params = {
            "n_epochs_warmup": 0,
            "latent_lr": varying_arg["latent_lr"],
            "latent_wd": varying_arg["latent_wd"],
            "attribute_nn_lr": varying_arg["attribute_nn_lr"],
            "attribute_nn_wd": varying_arg["attribute_nn_wd"],
            "step_size_lr": varying_arg["step_size_lr"],
            "cosine_scheduler": varying_arg["cosine_scheduler"],
            "scheduler_final_lr": varying_arg["scheduler_final_lr"],
            "decoder_lr": varying_arg["decoder_lr"],
            "decoder_wd": varying_arg["decoder_wd"]
        }

    def validate_config(self):
        # TO DO - CHECK HAVE ALL REQUIRED ARGS
        pass

    @staticmethod
    def preprocess(
        adata: anndata.AnnData, 
        config : dict,
        data_name : str,
        outs_dir : str,
        pert_column : str,
        control_key : str,
    ) -> anndata.AnnData:
        # Assign variables
        cores = os.cpu_count()
        adata.obs["condition"] = adata.obs[pert_column].astype(str)
        adata.obs["condition_name"] = adata.obs["condition"].astype(str)
        adata.obs.loc[~(adata.obs['condition'] == control_key), 'condition_name'] = adata.obs.loc[~(adata.obs['condition'] == control_key), 'condition_name'] + '_pert'
        adata.obs["condition"] = adata.obs["condition"].astype('category')
        
        
        # generate importance scores from go ontology
        pert_list = list(adata.obs[pert_column].unique())
        gene_list = list(adata.var_names)
        
        pert_list.remove(control_key) if control_key in pert_list else None

        df = make_GO(outs_dir, pert_list, data_name, num_workers=cores, save=True)
        df = df[df['source'].isin(gene_list)]
        
        
        # create pert2neighbor
        pert2neighbor =  {i: get_map(i) for i in list(adata.obs[pert_column].cat.categories)}
        adata.uns["pert2neighbor"] = pert2neighbor
        
        
        # get importance scores relative to pertubations of interest
        pert2neighbor = np.asarray([val for val in adata.uns["pert2neighbor"].values()])
        keep_idx = pert2neighbor.sum(0) > 0
        
        
        # create mapper and mean expression matrix
        name_map = dict(adata.obs[["condition", "condition_name"]].drop_duplicates().values)
        ctrl = np.asarray(adata[adata.obs["condition"].isin([control_key])].X.mean(0)).flatten()
        df_perts_expression = pd.DataFrame(adata.X.A, index=adata.obs_names, columns=adata.var_names)
        df_perts_expression["condition"] = adata.obs["condition"]
        df_perts_expression = df_perts_expression.groupby(["condition"]).mean()
        df_perts_expression = df_perts_expression.reset_index()

        
        #Currently not known how the format for multiple pertubations is.

        #Given this, assuming it will work in the following structure and make a "generalised" for loop instead of just boolean style which would work for only dealing with single pertubations.
        
        #Assumed structure data will be in:
        
        #- control : control
        #- PertA : single pertubation
        #- PertA+PertB : dual pertubation
        
        # loop through to define lists of pertubations and values
        single_pert_val = []
        single_perts_condition = []

        dual_pert_val = []
        dual_perts_condition = []

        for pert in adata.obs["condition"].cat.categories:

          # control
            if pert == control_key:
                pass

          # single pertubation
            elif "+" not in pert:
                single_pert_val.append(pert)
                single_perts_condition.append(f'{pert}_pert')

          # dual pertubation
            elif "+" in pert:
                dual_pert_val.append(pert)
                dual_perts_condition.append(f'{pert}_pert')

            else:
                raise NameError(f"The key {pert} doesn't form to the expected structure, please review value or structure!")

        # add control to single pert
        single_perts_condition.append(control_key)
        single_pert_val.append(control_key)
        
        
        # create the components for the anndata pertubations construction
        sliced_dict = {k: v for k, v in name_map.items() if v in single_perts_condition}
        df_singleperts_expression = pd.DataFrame(df_perts_expression.set_index("condition").loc[list(sliced_dict.keys())], index = single_pert_val)
        df_singleperts_condition = pd.Index(single_perts_condition)
        df_single_pert_val = pd.Index(single_pert_val)
        df_singleperts_emb = np.asarray([adata.uns["pert2neighbor"][p1][keep_idx] for p1 in df_singleperts_expression.index])
        
        adata_single = anndata.AnnData(X=df_singleperts_expression.values, var=adata.var.copy(), dtype=df_singleperts_expression.values.dtype)
        adata_single.obs_names = df_singleperts_condition
        adata_single.obs["condition"] = df_singleperts_condition
        adata_single.obs["perts_name"] = df_single_pert_val
        adata_single.obsm["perturbation_neighbors"] = df_singleperts_emb

        biolord.Biolord.setup_anndata(
                adata_single,
                ordered_attributes_keys=[varying_arg["ordered_attributes_key"]],
                categorical_attributes_keys=None,
                retrieval_attribute_key=None,
            )
        
        return adata_single
    

    def train(self):
        self.model.train(
            max_epochs=int(self.config["max_epochs"]),
            batch_size=32,
            plan_kwargs=self.trainer_params,
            early_stopping=True,
            early_stopping_patience=int(self.config["early_stopping_patience"]),
            check_val_every_n_epoch=5,
            num_workers=1,
            enable_checkpointing=False
        )

    # Outputs needed from preprocessing
    # self.adata where cells of each perturbation have been averaged 
    # should have split as obs key (where values can be train, val, test as string)

    def predict(self):
        # Define perturbations of interest
        # All perturbations
        perts = self.adata.obs["condition"].cat.categories

        # Pertubations which have been seen in training 
        train_condition_perts = self.adata[self.adata.obs[self.split_key] == 'train'].obs["condition"].cat.categories
        
        # Get control and perturbed datasets
        adata_control = self.adata[self.adata.obs["condition"] == "ctrl"]
        dataset_control = self.model.get_dataset(adata_control)

        dataset_reference = self.model.get_dataset(self.adata)

        output_dict = {}

        for i, pert in enumerate(perts):
            if pert in train_condition_perts:
                idx_ref =  bool2idx(self.adata_single.obs["condition"] == pert)[0]         
                # for seen perturbations, get true gene expression
                expression_pert = dataset_reference["X"][[idx_ref], :].mean(0).cpu().numpy()

                output_dict[f'{pert}_true'] = expression_pert

            elif "ctrl" in pert:
                # for control cells, predict expression for if they had been perturbed
                idx_ref =  bool2idx(self.adata_single.obs["condition"] == pert)[0]
                expression_pert = dataset_reference["X"][[idx_ref], :].mean(0).cpu().numpy()

                dataset_pred = dataset_control.copy()
                dataset_pred[self.config["ordered_attributes_key"]] = repeat_n(dataset_reference[self.config["ordered_attributes_key"]][idx_ref, :], dataset_pred.shape[0])
                test_preds, _ = self.model.module.get_expression(dataset_pred)

                output_dict[f'{pert}_pred'] = test_preds.cpu().numpy()


        # Prepare output
        # TO DO - WRANGLE FOR METRICS
        return output_dict




# TO IMPLEMENT

    # @classmethod
    # def load(self, adata, config):
    #     model = cpa.CPA.load(
    #         dir_path=config["path"]["save_path"],
    #         adata=adata,
    #         use_gpu=True
    #     )
    #     return model
        
    # def save(self):
    #     output_dir_path=self.config["path"]["output_dir_path"]
    #     output_file_name=self.config["path"]["output_file_name"]
    #     self.adata.write(os.path.join(output_dir_path, output_file_name))





