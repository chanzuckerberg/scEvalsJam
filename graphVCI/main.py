from .gvci.train import train


def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """

    # setting arguments

    args_dict = {
        "name": "default_run",
        "artifact_path": "./artifacts/",
        # "data_path": "../graphVCI/datasets/marson_prepped.h5ad",
        # "graph_path": "../graphVCI/graphs/marson_grn_128.pth",
        # "covariate_keys": "covariates",

        # "data_path": "../1gene-norman.h5ad",
        "graph_path": None,
        "covariate_keys": None,

        "cpu": "store_true",
        "gpu": "0",
        "seed": None,
        "batch_size": 128,
        "max_epochs": 2000,
        "patience": 20,
        "checkpoint_freq": 10,
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

    }

    return args_dict


def train_model(anndata, args):
    train(anndata, args)
