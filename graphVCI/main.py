from gvci.train import train

import argparse


def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """
    parser = argparse.ArgumentParser()

    # setting arguments

    args_dict = {
        "name": "default_run",
        "artifact_path": "./artifacts/",
        "data_path": "./datasets/marson_prepped.h5ad",
        "graph_path": "./graphs/marson_grn_128.pth",
        "covariate_keys": "covariates",

        # "data_path": "../1gene-norman.h5ad",
        # "graph_path": None,

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
        "hparams": "hparams.json",
        "omega0": 1.0,
        "omega1": 10.0,
        "omega2": 0.1,
        "graph_mode": "sparse",

    }

    return args_dict


def train_model(args):
    train(args)


if __name__ == "__main__":
    train(parse_arguments())
