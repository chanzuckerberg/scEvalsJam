from perturbench import PerturbationModel
from perturbench import PerturbationDataset
import cpa


class CPA(PerturbationModel):
    def __init__(self, adata, config):
        self.config = config
        self.adata = self.preprocess(adata)
        self.model = cpa.CPA(
            adata=self.adata, 
            split_key='split',
            train_split='train',
            valid_split='val',
            test_split='test',
            **self.config["model_params"],
        )

    @staticmethod
    def preprocess(adata):
        adata.obs["perturbation_name"].replace({"control": "ctrl"}, inplace=True)
        adata.X = adata.layers['counts'].copy()
        cpa.CPA.setup_anndata(
            adata, 
            perturbation_key='perturbation_name',
            control_group='ctrl',
            # dosage_key='dose_value',
            # categorical_covariate_keys=['cell_type'],
            is_count_data=True,
            # deg_uns_key='rank_genes_groups_cov',
            # deg_uns_cat_key='cov_cond',
            max_comb_len=2,
        )
        # adata.obs['split'] = np.random.choice(['train', 'valid'], size=adata.n_obs, p=[0.85, 0.15])
        # adata.obs.loc[adata.obs['perturbation_name'].isin(['DUSP9+ETS2', 'CBL+CNN1']), 'split'] = 'ood'
        return adata

    def train(self):
        self.model.train(
            max_epochs=1,
            use_gpu=True, 
            batch_size=2048,
            plan_kwargs=self.config["trainer_params"],
            early_stopping_patience=5,
            check_val_every_n_epoch=5,
            save_path=self.config["path"]["save_path"],
        )

    def predict(self):
        self.adata.layers['X_true'] = self.adata.X.copy()
        ctrl_adata = self.adata[self.adata.obs['perturbation_name'] == 'ctrl'].copy()
        adata.X = ctrl_adata.X[np.random.choice(ctrl_adata.n_obs, size=adata.n_obs, replace=True), :]
        self.model.predict(self.adata, batch_size=2048)
        self.adata.layers['CPA_pred'] = self.adata.obsm['CPA_pred'].copy()
        
    
    def load(self, adata):
        model = cpa.CPA.load(
            dir_path=self.config["path"]["save_path"],
            adata=adata,
            use_gpu=True
        )
        return model