from models.model import PerturbationModel
from dataset import PerturbationDataset
from scvi.train import Trainer
import anndata as ad
import scipy as sp
import pathlib
import torch
import scgen



class ScGenModel(PerturbationModel):
    def __init__(
        self, 
        adata_train: ad.AnnData,
        batch_key: str,
        labels_key: str,
        ctrl_key : str,
        stim_key : str,
        cond_to_predict : str,
        save_model_path: pathlib.Path,
        device: torch.cuda.device,
        **kwargs
        ) -> None:
        self.adata_train = adata_train
        self.adata_test = None
        self.batch_key = batch_key
        self.ctrl_key = ctrl_key
        self.stim_key = stim_key
        self.cond_to_predict = cond_to_predict
        self.save_model_path = save_model_path
        self.device = device
        self.train_params = None
        self.trained = False
        self.model_name = 'scGen'
        self.pred = None
        self.delta = None

        # Preprocess adata
        scgen.SCGEN.setup_anndata(
            self.adata_train,
            batch_key=batch_key,
            labels_key=labels_key)

        # Update model params
        # self.model_params = {k:v for k,v in kwargs.items() if k in list(scgen.SCGEN.__init__.__code__.co_varnames) + list(scgen.SCGENVAE.__init__.__code__.co_varnames)}
        # self.model_params.update({'adata':self.adata})

        # Initialise model
        # self.model = scgen.SCGEN(
        #     **self.model_params
        # )
        self.model = scgen.SCGEN(self.adata_train)


    def train(self, train_data: PerturbationDataset, **kwargs) -> None:        
        # Define training params
        self.train_params = {'max_epochs':5,
                             'batch_size':32,
                             'early_stopping':True,
                             'early_stopping_patience':5}
        if len(kwargs) != 0:
            new_train_params = {k:v for k,v in kwargs.items() if k in list(Trainer.__init__.__code__.co_varnames)}
            self.train_params = self.train_params.update(new_train_params)
        
        # Train model
        self.model.train(
            **self.train_params
        )
        self.trained = True

    def predict(self, test_data: PerturbationDataset, 
                pertturbation_list = [], **kwargs):
        
        self.adata_test = test_data.anndata()
        print('self.batch_key: ', self.batch_key)
        print('self.ctrl_key: ', self.ctrl_key)
        
        print(self.adata_test.obs.groupby([self.batch_key, self.batch_key]).size())
        self.adata_test = self.adata_test[
            self.adata_test.obs[self.batch_key] == self.ctrl_key]
        print('train_data:')
        print(self.adata_train)
        print('before predict:')
        print(self.adata_test)
        pred, delta = self.model.predict(
            ctrl_key=self.ctrl_key,
            stim_key=self.stim_key,
            # celltype_to_predict=self.cond_to_predict,
            adata_to_predict=self.adata_test
        )
        pred.obs['condition'] = 'pred'
        assert pred.shape[0] == self.adata_test.shape[0]
        
        self.pred = pred
        self.delta = delta

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
