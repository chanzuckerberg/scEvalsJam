"""
Training script for SAMS-VAE and baseline models

Single run:
python train.py --config {config_path}
See configs/example.yaml for example config
Results can be logged locally or to Weights and Biases

WandB sweep:
wandb sweep {sweep_config_path}
wandb agent {sweep_id}
See configs/example_sweep.yaml for example config
Requires wandb account
"""
import argparse
import os
from collections import defaultdict
from os.path import exists, join
from typing import Any, DefaultDict, Dict

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from sams_vae import data
from sams_vae.data.utils.perturbation_datamodule import PerturbationDataModule
from sams_vae.models.utils.lightning_callbacks import (
    GradientNormTracker,
    TreatmentMaskStatsTracker,
)
from sams_vae.models.utils.perturbation_lightning_module import (
    TrainConfigPerturbationLightningModule,
)

RESULTS_BASE_DIR = "results/"


def train(config: Dict):
    # if launched as part of wandb sweep, will have empty config
    wandb_sweep = len(config) == 0
    if wandb_sweep:
        # launched as part of WandB sweep
        # config is retrieved from WandB, so need to preprocess after
        # initializing experiment
        results_dir, logger, wandb_run, config = init_experiment_wandb(config)
        config = preprocess_config(config)
    elif config["use_wandb"]:
        # local experiment logging to WandB
        config = preprocess_config(config)
        results_dir, logger, wandb_run, config = init_experiment_wandb(config)
    else:
        # local experiment logging to results/
        config = preprocess_config(config)
        results_dir, logger = init_experiment_local(config)
        wandb_run = None

    data_module = get_data_module(config)

    # adds data dimensions, statistics for initialization of model / guide
    config = add_data_info_to_config(config, data_module)

    pl.seed_everything(config["seed"])

    lightning_module = TrainConfigPerturbationLightningModule(
        config=config,
        D_obs_counts_train=data_module.get_train_perturbation_obs_counts(),
        D_obs_counts_val=data_module.get_val_perturbation_obs_counts(),
        D_obs_counts_test=data_module.get_test_perturbation_obs_counts(),
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    callbacks, best_checkpoint_callback = get_callbacks(results_dir, data_module)

    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=logger,
        callbacks=callbacks,
        max_epochs=config.get("max_epochs"),
        max_steps=config.get("max_steps", -1),
        gradient_clip_val=config.get("gradient_clip_norm"),
    )

    trainer.fit(
        lightning_module,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

    # load the best checkpoint and save validation metrics
    checkpoint_path = best_checkpoint_callback.best_model_path
    lightning_module = TrainConfigPerturbationLightningModule.load_from_checkpoint(
        checkpoint_path
    )

    # TODO: clean up
    if lightning_module.predictor is not None:
        val_iwelbo = lightning_module.predictor.compute_predictive_iwelbo(
            data_module.val_dataloader(), n_particles=100
        )["IWELBO"].mean()
    else:
        val_iwelbo = None

    if wandb_run is not None:
        wandb_run.summary["val/IWELBO"] = val_iwelbo
        wandb_run.summary["best_checkpoint_path"] = checkpoint_path
    else:
        summary_df = pd.DataFrame(
            {"val/IWELBO": [val_iwelbo], "best_checkpoint_path": [checkpoint_path]}
        )
        summary_df.to_csv(join(results_dir, "summary.csv"), index=False)

    return val_iwelbo


def init_experiment_local(config: Dict):
    # save locally to results directory
    # enumerates subdirectories in case of repeated runs
    results_dir = join(RESULTS_BASE_DIR, config["name"])
    i = 2
    while exists(results_dir):
        results_dir = join(RESULTS_BASE_DIR, f"{config['name']}-{i}")
        i += 1
    os.makedirs(results_dir)

    logger = CSVLogger(results_dir)
    return results_dir, logger


def init_experiment_wandb(config: Dict):
    kwargs = config.get("wandb_kwargs", dict())
    run = wandb.init(config=config, **kwargs)
    # if part of wandb sweep, run.config has values assigned through sweep
    # otherwise, returns same values from initialization
    config = run.config.as_dict()
    results_dir = run.dir
    logger = WandbLogger()
    return results_dir, logger, run, config


def get_data_module(config: Dict) -> PerturbationDataModule:
    kwargs = config.get("data_module_kwargs", dict())
    data_module = getattr(data, config["data_module"])(**kwargs)
    return data_module


def preprocess_config(config: Dict):
    config_v2 = {}
    # allow specification of variables that share values by
    # by connecting with "--" (useful for wandb sweeps)
    for k in config.keys():
        if "--" in k:
            val = config[k]
            new_keys = k.split("--")
            for new_key in new_keys:
                config_v2[new_key] = val
        else:
            config_v2[k] = config[k]

    processed_config: DefaultDict[str, Any] = defaultdict(dict)
    # convert from . notation to nested
    # eg config["model_kwargs.n_latent"] -> config["model_kwargs"]["n_latent"]
    # only needs to support single layer of nesting for now TODO: clean up
    for k, v in config_v2.items():
        if "." not in k:
            processed_config[k] = v
        else:
            split_k_list = k.split(".", maxsplit=1)
            processed_config[split_k_list[0]][split_k_list[1]] = v

    # allow specification of shared n_latent
    if "n_latent" in processed_config:
        processed_config["model_kwargs"]["n_latent"] = processed_config["n_latent"]
        processed_config["guide_kwargs"]["n_latent"] = processed_config["n_latent"]

    # replace -1 in gradient_clip_norm with None
    if processed_config.get("gradient_clip_norm") == -1:
        processed_config["gradient_clip_norm"] = None

    print(processed_config)

    return processed_config


def add_data_info_to_config(config: Dict, data_module: PerturbationDataModule):
    # insert data dependent fields to config
    if "model_kwargs" not in config:
        config["model_kwargs"] = dict()

    if "guide_kwargs" not in config:
        config["guide_kwargs"] = dict()

    config["model_kwargs"]["n_treatments"] = data_module.get_d_var_info().shape[0]
    config["model_kwargs"]["n_phenos"] = data_module.get_x_var_info().shape[0]

    config["guide_kwargs"]["n_treatments"] = data_module.get_d_var_info().shape[0]
    config["guide_kwargs"]["n_phenos"] = data_module.get_x_var_info().shape[0]
    config["guide_kwargs"][
        "x_normalization_stats"
    ] = data_module.get_x_train_statistics()

    return config


def get_callbacks(results_dir: str, data_module: PerturbationDataModule):
    checkpoint_dir = join(results_dir, "checkpoints")
    # store checkpoint with the best validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        auto_insert_metric_name=False,
        filename="best-epoch={epoch}-step={step}-val_loss={val/loss:.2f}",
    )
    # save a checkpoint 2000 training steps TODO: change frequency?
    checkpoint_callback_2 = ModelCheckpoint(
        dirpath=checkpoint_dir,
        every_n_train_steps=2000,
        auto_insert_metric_name=False,
        filename="epoch={epoch}-step={step}-val_loss={val/loss:.2f}",
    )
    gradient_norm_callback = GradientNormTracker()
    mask_stats_callback = TreatmentMaskStatsTracker(
        mask_key="mask",
        true_latent_effects=data_module.get_simulated_latent_effects(),
        d_var=data_module.get_d_var_info(),
    )
    callbacks = [
        checkpoint_callback,
        checkpoint_callback_2,
        gradient_norm_callback,
        mask_stats_callback,
    ]
    return callbacks, checkpoint_callback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Experiment configuration yaml path, "
        "required if not running as part of wandb sweep",
        default=None,
    )

    # parse known args only because wandb sweep adds command line arguments
    args, unknown = parser.parse_known_args()

    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        # if part of wandb sweep, will fill in config
        # with hyperparameters assigned as part of sweep
        config = dict()

    train(config)
