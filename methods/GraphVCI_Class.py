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

    def eval(self, anndata, perts, graph_path=None, model=None):
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

        # Only use control cells
        ctl_mask = anndata.obs['perturbation_name'] == "control"

        # TODO: filtering
        self._process_anndata(anndata)

        ctl_datasets = load_dataset_splits(
            anndata, covariate_keys=None, test_ratio=None,
            sample_cf=True,
        )['training']

        # Extract data
        genes_control = ctl_datasets.genes[ctl_mask]
        perts_control = ctl_datasets.perturbations[ctl_mask]
        pert_vector = ctl_datasets.perts_dict[perts]
        pert_vectors = pert_vector.repeat(genes_control.size(0), 1)
        covars_control = [torch.ones([genes_control.size(0), 1])]  # Currently no fixed covariates

        # print(f'{genes_control.shape = }, {perts_control.shape = }, {pert_vectors.shape = }, {covars_control[0].shape = }')
        # print(covars_control[0].shape)
        out = self.model.predict(
            genes_control,
            perts_control,
            pert_vectors,
            [covar for covar in covars_control]
        )

        out_anndata = anndata[anndata.obs['perturbation_name'] == "control"].copy()
        out_anndata.X = out

        # print(out_anndata.X)
        # print(out_anndata.X.shape)

        return out_anndata

    def _process_anndata(self, anndata):
        """ Process anndata, used for training and testing """
        sc.pp.normalize_total(anndata)
        sc.pp.highly_variable_genes(anndata, n_top_genes=123, subset=True)


if __name__ == "__main__":
    gcvi = GraphVCI_ABC()
    import scanpy as sc

    # anndata = sc.read("../graphVCI/graphs/marson_grn_128.pth")
    _anndata = sc.read("../1gene-norman.h5ad")
    _anndata = my_process_adata(_anndata)

    # gcvi.train(anndata)

    model_save = torch.load("/home/maccyz/Documents/scEvalsJam/methods/artifacts2/saves/default_run_2024.06.14_16:12:13/model_seed=None_epoch=0.pt")
    gcvi.eval(_anndata, "control", None, model_save)
