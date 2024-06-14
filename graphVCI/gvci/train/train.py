import os
import time
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from ..model import load_graphVCI

from vci.dataset import load_dataset_splits
from vci.evaluate import evaluate, evaluate_classic
from vci.utils.general_utils import initialize_logger, ljson
from vci.utils.data_utils import data_collate
from ..utils.graph_utils import get_graph


def prepare(anndata, graph_data, args, covariate_keys=None, state_dict=None):
    """
    Instantiates autoencoder and dataset to run an experiment.
    """

    datasets = load_dataset_splits(
        anndata, covariate_keys=covariate_keys,
        sample_cf=(True if args["dist_mode"] == 'match' else False),
    )

    args["num_outcomes"] = datasets["training"].num_genes
    args["num_treatments"] = datasets["training"].num_perturbations
    args["num_covariates"] = datasets["training"].num_covariates

    # Generate graph
    if args['graph_mode'] == "dense":  # row target, col source
        output_adj_mode = "target_to_source"
    elif args['graph_mode'] == "sparse":  # first row souce, second row target
        output_adj_mode = "source_to_target"
    else:
        ValueError("graph_mode not recognized")
    if type(graph_data) == str:
        graph_data = torch.load(graph_data)
    else:
        graph_data = None
    node_features, adjacency, edge_features = get_graph(graph=graph_data,
        n_nodes=args['num_outcomes'], n_features=args["graph_latent_dim"],
        graph_mode=args["graph_mode"], output_adj_mode=output_adj_mode,
        add_self_loops=True)
    graph_data = (node_features, adjacency, edge_features)

    model = load_graphVCI(graph_data, args, state_dict)

    return model, datasets


def train(anndata, graph_data, args):
    """
    Trains a graphVCI model
    """
    if args["seed"] is not None:
        torch.manual_seed(args["seed"])

    model, datasets = prepare(anndata, graph_data, args, args['covariate_keys'])

    datasets.update(
        {
            "loader_tr": torch.utils.data.DataLoader(
                datasets["training"],
                batch_size=args["batch_size"],
                shuffle=True,
                collate_fn=(lambda batch: data_collate(batch, nb_dims=1))
            )
        }
    )

    args["hparams"] = model.hparams

    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    writer = SummaryWriter(log_dir=os.path.join(args["artifact_path"], "runs/" + args["name"] + "_" + dt))
    save_dir = os.path.join(args["artifact_path"], "saves/" + args["name"] + "_" + dt)
    print(f'{save_dir = }')
    os.makedirs(save_dir, exist_ok=True)

    initialize_logger(save_dir)
    ljson({"training_args": args})
    ljson({"model_params": model.hparams | model.g_hparams})
    logging.info("")

    start_time = time.time()
    for epoch in range(args["max_epochs"]):
        print(f'{epoch = }')
        epoch_training_stats = defaultdict(float)

        for data in datasets["loader_tr"]:
            (genes, perts, cf_genes, cf_perts, covariates) = (
                data[0], data[1], data[2], data[3], data[4:])

            minibatch_training_stats = model.update(
                genes, perts, cf_genes, cf_perts, covariates
            )

            for key, val in minibatch_training_stats.items():
                epoch_training_stats[key] += val
        model.update_eval_encoder()

        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val / len(datasets["loader_tr"])
            if not (key in model.history.keys()):
                model.history[key] = []
            model.history[key].append(epoch_training_stats[key])
        model.history["epoch"].append(epoch)

        ellapsed_minutes = (time.time() - start_time) / 60
        model.history["elapsed_time_min"] = ellapsed_minutes

        # decay learning rate if necessary
        # also check stopping condition: 
        # patience ran out OR max epochs reached
        stop = (epoch == args["max_epochs"] - 1)

        if (epoch % args["checkpoint_freq"]) == 0 or stop:
            if args["eval_mode"] == "native":
                evaluation_stats = evaluate(
                    model, datasets,
                    batch_size=args["batch_size"]
                )
            elif args["eval_mode"] == "classic":
                evaluation_stats = evaluate_classic(
                    model, datasets,
                    batch_size=args["batch_size"]
                )
            else:
                raise ValueError("eval_mode not recognized")

            for key, val in evaluation_stats.items():
                if not (key in model.history.keys()):
                    model.history[key] = []
                model.history[key].append(val)
            model.history["stats_epoch"].append(epoch)

            ljson(
                {
                    "epoch": epoch,
                    "training_stats": epoch_training_stats,
                    "evaluation_stats": evaluation_stats,
                    "ellapsed_minutes": ellapsed_minutes,
                }
            )

            for key, val in epoch_training_stats.items():
                writer.add_scalar(key, val, epoch)

            print("Saving Model")
            torch.save(
                (model.state_dict(), args, model.history),
                os.path.join(
                    save_dir,
                    "model_seed={}_epoch={}.pt".format(args["seed"], epoch),
                ),
            )

            ljson(
                {
                    "model_saved": "model_seed={}_epoch={}.pt\n".format(
                        args["seed"], epoch
                    )
                }
            )
            stop = stop or model.early_stopping(np.mean(evaluation_stats["test"]))
            if stop:
                ljson({"early_stop": epoch})
                break

    writer.close()
    return model
