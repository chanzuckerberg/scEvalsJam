import torch 
from gears import PertData, GEARS
import scanpy as sc

class Abstract:
    def __init__(self, model):
        pass



class Gears_ABC(Abstract):
    def __init__(self, model=None):

        if model is None:
            # initiate gears from scratch
            self.mode = None
        else:
            # we can load in the pretrained model
            if isinstance(model, str):
                self.model = model
            


    def train(self, anndata, split_obs, 
                dataset_kws:dict={'batch_size'}, 
                model_kws:dict={}, 
                train_kws:dict={}):
        """
        call the training step of cell-gears

        Arguments:
        ----
        anndata : the processed ?/ raw anndata object
        split_obs : the obs column name used as to split the train-test
        dataset_kws : dict, the keyword arguments to pass to pert_adat.prepare, including #TODO: add args
        model_kws : dict, the keyword arguments to pass to model architecture, including #TODO: add args
        train_kws : dcit, the keyword arguments to control the training process, including #TODO: add args

        Return:
        ---
        a trained model 
        """
        # raw count data in

        sc.pp.normalize_total(anndata)
        sc.pp.log1p(anndata)
        sc.pp.highly_variable_genes(anndata,n_top_genes=5000, subset=True)
        
        
        # prepare data
        batch_size = dataset_kws['batch_size']
        

        pert_data = PertData('../datasets') # specific saved folder
        pert_data.new_data_process(dataset_name = dataset_kws['name'], adata = anndata) # specific dataset name and adata object
        pert_data.load(data_path = f'../datasets/{dataset_kws['name']}') # load the processed data, the path is saved folder + dataset_name
        pert_data.prepare_split(split = split_obs, seed = 1) # get data split with seed
        pert_data.get_dataloader(batch_size = batch_size, test_batch_size = batch_size) # prepare data loader
        

        #trainig

        # config the training params
        if "device" not in train_kws:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            device = train_kws['device']
            train_kws.pop('deivce');

        if self.model is None:
            gears_model = GEARS(pert_data, device = device, 
                        weight_bias_track = False, 
                        proj_name = 'pertnet', 
                        exp_name = 'pertnet')
            gears_model.model_initialize(**model_kws)
            gears_model.train(**train_kws)
            self.model = gears_model
            gears_model.save_model('gears.ckpt')

        else:
            gears_model = GEARS(pert_data, device = device, 
                        weight_bias_track = False, 
                        proj_name = 'pertnet', 
                        exp_name = 'pertnet')
            gears_model.load(self.model)
            self.model = gears_model
            gears_model.save_model('gears.ckpt')



