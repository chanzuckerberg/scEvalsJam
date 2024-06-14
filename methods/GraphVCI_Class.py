import graphVCI.gvci.train.train as train_model
import torch
from vci.dataset import load_dataset_splits
from graphVCI.gvci.model import load_graphVCI


def my_process_adata(adata):
    """ Processing for Norman dataset """
    print("Adjusting dataset for Norman")
    # Fields
    fields = {}
    adata.uns['fields'] = fields

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

    def train(self, anndata, model_kws: dict = None):
        if model_kws is None:
            model_kws = {
                "name": "default_run",
                "artifact_path": "./artifacts2/",
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

        # TODO: is filtering needed?
        self._process_anndata(anndata)

        graph_path = None
        train_model(anndata, graph_path, args=model_kws)


    def eval(self, ctl_anndata, perts, graph_path=None, model=None):
        if self.model is None and model is None:
            raise ValueError("Model not trained yet")

        # Load model, including origial graph
        if self.model is None:
            state_dict, args, _ = model
            adjacency = state_dict['adjacency']
            edge_features = state_dict['edge_features']
            node_features = state_dict['node_features']
            graph_data = (node_features, adjacency, edge_features)

            self.model = load_graphVCI(graph_data, args, state_dict=state_dict)

        # Process model

        # TODO: filtering
        self._process_anndata(ctl_anndata)

        ctl_datasets = load_dataset_splits(
            ctl_anndata, covariate_keys=None, test_ratio=None,
            sample_cf=True,
        )['training']

        genes_control = ctl_datasets.genes
        perts_control = ctl_datasets.perturbations
        pert_vector = ctl_datasets.perts_dict[perts]
        pert_vectors = pert_vector.repeat(genes_control.size(0), 1)
        covars_control = [torch.ones([genes_control.size(0), 1])]  # Currently no fixed covariates

        # print(covars_control[0].shape)
        out = self.model.predict(
            genes_control[:10],
            perts_control[:10],
            pert_vectors[:10],
            [covar[:10] for covar in covars_control]
        )
        print(out)
        print(out.shape)

    def _process_anndata(self, anndata):
        """ Process anndata, used for training and testing """
        sc.pp.normalize_total(anndata)
        sc.pp.highly_variable_genes(anndata, n_top_genes=123, subset=True)



if __name__ == "__main__":
    gcvi = GraphVCI_ABC()
    import scanpy as sc

    # anndata = sc.read("../graphVCI/graphs/marson_grn_128.pth")
    anndata = sc.read("../1gene-norman.h5ad")
    anndata = my_process_adata(anndata)

    # gcvi.train(anndata)

    model_save = torch.load("/home/maccyz/Documents/scEvalsJam/methods/artifacts2/saves/default_run_2024.06.14_16:12:13/model_seed=None_epoch=0.pt")
    gcvi.eval(anndata, "control", None, model_save)
