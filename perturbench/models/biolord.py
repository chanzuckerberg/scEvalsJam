from perturbench import PerturbationModel
import numpy as np
import os
import biolord
from ..utils import (
    make_GO, get_map, bool2idx, repeat_n
)
import anndata as ad


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
    def preprocess(adata, config):
        # adata.obs["perturbation_name"].replace({"control": "ctrl"}, inplace=True)
        # adata.X = adata.layers['counts'].copy()
        
        ordered_attributes_key = config["ordered_attributes_key"]

        # generate importance scores from go ontology

        # Set up variables
        cores = os.cpu_count()
        pert_list = list(adata.obs[config['pert_column']].unique())
        gene_list = list(adata.var_names)
        pert_column = config['pert_column']

        df = make_GO(config['data_path'], pert_list, config['model_name'], num_workers=cores, save=True)
        df = df[df['source'].isin(gene_list)]

        # create pert2neighbor
        pert2neighbor =  {i: get_map(i) for i in list(adata.obs[pert_column].cat.categories)}
        adata.uns["pert2neighbor"] = pert2neighbor

        # get importance scores relative to pertubations of interest
        pert2neighbor = np.asarray([val for val in adata.uns["pert2neighbor"].values()])
        keep_idx = pert2neighbor.sum(0) > 0


        # ... missing here
        # unclear how much needs to be here vs unified


        biolord.Biolord.setup_anndata(
                adata,
                ordered_attributes_keys=[ordered_attributes_key],
                categorical_attributes_keys=None,
                retrieval_attribute_key=None,
            )
        
        return adata
    

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





