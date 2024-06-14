import torch 
import os
import pathlib
from gears import PertData, GEARS
import scanpy as sc
from perturbench import PerturbationModel
from perturbench import PerturbationDataset


class GearsModel(PerturbationModel):
    def __init__(self, device, model=None, data_dir=None):

        super().__init__(device)

        self.device = device
        if model is None:
            # initiate gears from scratch
            self.model = None
        else:
            # we can load in the pretrained model
            if isinstance(model, str):
                self.model = model

        if data_dir is None:
            self.data_dir = "../datasets"
            print("Gears is a graph neural network that requires the dataset to initialize the model")
            print("the data directory is set to default : ", self.data_dir)
        else:
            self.data_dir = data_dir
        
    

    def create_pertdata(self, data, split_mode, dataset_kws):
        
        anndata = data.anndata if isinstance(data, PerturbationDataset) else data

        # normalizes
        if 'highly_variable' not in anndata.var_keys():
            sc.pp.normalize_total(anndata)
            sc.pp.log1p(anndata)
            sc.pp.highly_variable_genes(anndata,n_top_genes=5000, subset=True)
        
        

        # prepare data
        pert_data = PertData(self.data_dir) # specific saved folder
        pert_data.new_data_process(dataset_name = dataset_kws['name'], adata = anndata) # specific dataset name and adata object
        
        return self.load_data(split_mode, dataset_kws)

    def load_data(self, split_mode, dataset_kws):
        

        batch_size = dataset_kws['batch_size'] if 'batch_size' in dataset_kws else 32

        pert_data = PertData(self.data_dir) # specific saved folder
        pert_data.load(data_path = os.path.join(self.data_dir, dataset_kws['name'])) # load the processed data, the path is saved folder + dataset_name
        pert_data.prepare_split(split = split_mode, seed = 1) # get data split with seed
        pert_data.get_dataloader(batch_size = batch_size, test_batch_size = batch_size) # prepare data loader

        self.pert_data = pert_data

        
        return pert_data

    def train(self, data, split_mode, 
                dataset_kws:dict={'batch_size':32, 'name':'test_data'}, 
                model_kws:dict={}, 
                train_kws:dict={}):
        """
        call the training step of cell-gears

        Arguments:
        ----
        anndata : the processed ?/ raw anndata object
        split_mode :  'simulation', 'simulation_single', 'combo_seen0', 
                    'combo_seen1', 'combo_seen2',
                    'single', 'no_test', 'no_split', 'custom'
        dataset_kws : dict, the keyword arguments to pass to pert_adat.prepare, including #TODO: add args
        model_kws : dict, the keyword arguments to pass to model architecture, including #TODO: add args
        train_kws : dcit, the keyword arguments to control the training process, including #TODO: add args

        Return:
        ---
        a trained model 
        """
        # raw count data in

        if not os.path.exists(os.path.join(self.data_dir, dataset_kws['name'])):
            print("preprocess the data from raw count")
            self.create_pertdata(data, split_mode, dataset_kws)
        else:
            self.load_data(split_mode, dataset_kws);
        

        
        if self.model is None:
            # if provide a model checkpoint

            gears_model = GEARS(self.pert_data, device = self.device, 
                        weight_bias_track = False, 
                        proj_name = 'pertnet', 
                        exp_name = 'pertnet')
            gears_model.model_initialize(**model_kws)
            gears_model.train(**train_kws)
            self.model = gears_model

            # save the model
            gears_model.save_model(f"{dataset_kws['name']}_gears.ckpt")

        else:
            # if provide a model checkpoint

            gears_model = GEARS(self.pert_data, device = self.device, 
                        weight_bias_track = False, 
                        proj_name = 'pertnet', 
                        exp_name = 'pertnet')
            gears_model.load(self.model)
            self.model = gears_model

            # save the model
            gears_model.save_model(f"{dataset_kws['name']}_gears.ckpt")



    def predict(self,perturbation: list):
        """
        predict the 


        """
        if self.model is None:
            raise ValueError("the GEARS model is not defined")
        
        return self.model.predict(perturbation)
        
    
    def save(self, path: pathlib.Path) -> pathlib.Path:
        self.model.save_model(path)
        return 

    def load(self, path: pathlib.Path, device:str = 'cuda:0') -> None:

        try:
            self.pert_data
        except NameError:
            
            print("Gears is a graph neural network that requires the dataset to initialize the model")
            print("Please run `model.load_data` method firsts")


        # define grears with pert_data
        gears_model = GEARS(self.pert_data, device = device, 
                    weight_bias_track = False, 
                    proj_name = 'pertnet', 
                    exp_name = 'pertnet')
        
        gears_model.load(path)

        self.model = gears_model

            
            