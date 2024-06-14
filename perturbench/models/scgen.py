import sys
from perturbench.models import PerturbationModel
from perturbench.dataset import PerturbationDataset
import pathlib
import scipy as sp
import torch
import scgen
import anndata as ad
from scvi.train import Trainer


class ScGenModel(PerturbationModel):
    def __init__(
        self, 
        train_set: ad.AnnData,
        batch_key: str,
        labels_key: str,
        ctrl_key : str,
        stim_key : str,
        cond_to_predict : str,
        save_model_path: pathlib.Path,
        device: torch.cuda.device,
        **kwargs
        ) -> None:
        self.ctrl_key = ctrl_key
        self.stim_key = stim_key
        self.cond_to_predict = cond_to_predict
        self.save_model_path = save_model_path
        self.device = device
        self.train_params = None

        # Preprocess adata
        scgen.SCGEN.setup_anndata(train_set, batch_key=batch_key, labels_key=labels_key)

        # Update model params
        # self.model_params = {k:v for k,v in kwargs.items() if k in list(scgen.SCGEN.__init__.__code__.co_varnames) + list(scgen.SCGENVAE.__init__.__code__.co_varnames)}
        # self.model_params.update({'adata':self.adata})

        # Initialise model
        # self.model = scgen.SCGEN(
        #     **self.model_params
        # )
        self.model = scgen.SCGEN(train_set)


    def train(self, **kwargs) -> None:
        
        # Define training params
        self.train_params = {'max_epochs':100,
                             'batch_size':32,
                             'early_stopping':True,
                             'early_stopping_patience':25}
        if len(kwargs) != 0:
            new_train_params = {k:v for k,v in kwargs.items() if k in list(Trainer.__init__.__code__.co_varnames)}
            self.train_params = self.train_params.update(new_train_params)
        
        # Train model
        self.model.train(
            **self.train_params
        )

    def predict(self, **kwargs):
        pred, delta = self.model.predict(
            ctrl_key=self.ctrl_key,
            stim_key=self.stim_key,
            celltype_to_predict=self.cond_to_predict,
        )
        pred.obs['condition'] = 'pred'

        return pred, delta

    def save(self) -> pathlib.Path:
        self.model.save(self.save_model_path, overwrite=True)

    def load(self, path: pathlib.Path) -> None:
        if self.device == 'cuda':
            gpu_status=True
        else:
            gpu_status=False

        scgen.SCGEN.load(
            path, adata=self.adata, 
            use_gpu=gpu_status, prefix=None, 
            backup_url=None
            )