import graphVCI.gvci.train.train as train_model


def my_process_adata(adata):
    """ Processing for Norman dataset """
    print("Adjusting dataset for Norman")
    # Fields
    fields = {}
    adata.uns['fields'] = fields

    # # Randomly select a subset of genes to measure to reduce size
    # num_cols = min(adata.var.shape[0], 5000)
    # sampled_columns = np.random.choice(adata.var.shape[0], num_cols, replace=False)
    # adata = adata[:, sampled_columns]

    # Perturbation column name
    fields['perturbation'] = 'perturbation_name'

    # Control value
    adata.obs['control'] = (adata.obs['perturbation_name'] == 'control')

    # Set dose to 1
    adata.obs['dose'] = 1.0

    return adata


class GraphVCI_ABC:
    def __init__(self, model=None):
        if model is None:
            self.model = None
        else:
            pass

    def train(self, anndata, model_kws: dict=None):
        if model_kws is None:
            model_kws = {
                "name": "default_run",
                "artifact_path": "./artifacts/",
                # "graph_path": None,

                # "covariate_keys": "covariates",
                "covariate_keys": None,

                "cpu": "store_true",
                "gpu": "0",
                "seed": None,
                "batch_size": 128,
                "max_epochs": 2000,
                "patience": 20,
                "checkpoint_freq": 1,
                "eval_mode": "native",
                "outcome_dist": "normal",
                "dist_mode": "match",
                "encode_aggr": "sum",
                "decode_aggr": "att",
                "omega0": 1.0,
                "omega1": 10.0,
                "omega2": 0.1,
                "graph_mode": "sparse",
                "hparams": {
                    "latent_dim": 128,
                    "outcome_emb_dim": 256,
                    "treatment_emb_dim": 64,
                    "covariate_emb_dim": 16,
                    "encoder_width": 128,
                    "encoder_depth": 3,
                    "decoder_width": 128,
                    "decoder_depth": 3,
                    "discriminator_width": 64,
                    "discriminator_depth": 2,
                    "autoencoder_lr": 3e-4,
                    "discriminator_lr": 3e-4,
                    "autoencoder_wd": 4e-7,
                    "discriminator_wd": 4e-7,
                    "discriminator_steps": 3,
                    "step_size_lr": 45,
                },
                "graph_latent_dim": 128,

            }

        # Convert raw counts to normalised
        sc.pp.normalize_total(anndata)
        # sc.pp.log1p(anndata)
        sc.pp.highly_variable_genes(anndata, n_top_genes=100, subset=True)

        graph_path = None
        train_model(anndata, graph_path, args=model_kws)


if __name__ == "__main__":
    gcvi = GraphVCI_ABC()
    import scanpy as sc

    # anndata = sc.read("../graphVCI/graphs/marson_grn_128.pth")
    anndata = sc.read("../1gene-norman.h5ad")
    anndata = my_process_adata(anndata)

    gcvi.train(anndata)
