"""
Script to generate test set evaluation metrics from a training run

Usage:
python [experiment_path] [--wandb] [--perturbseq] [--batch_size {int}]

Examples:
    python eval.py michael/debug/mw545rhs --wandb --perturbseq
    - Saves evaluation metrics to wandb run summary

    python eval.py results/example --perturbseq
    - Saves evaluation metrics to results/example/test_metrics.csv (local experiment)

    python eval.py {checkpoint_path}.ckpt --perturbseq
    - Runs evaluation for specified checkpoint, and saves metrics to
      {checkpoint_path}_test_metrics.csv
"""

import argparse
import os
from os.path import basename, join, splitext
from typing import Any, Dict, Literal

import numpy as np
import pandas as pd
import torch
import wandb
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from sams_vae.data.utils.anndata import align_adatas
from sams_vae.models.utils.perturbation_lightning_module import (
    TrainConfigPerturbationLightningModule,
)


def evaluate_checkpoint(
    checkpoint_path: str,
    average_treatment_effect_method: Literal["mean", "perturbseq"],
    batch_size: int = 500,
    ate_n_particles: int = 2500,
) -> Dict[str, Any]:
    """
    Compute test set metrics for a given checkpoint


    Parameters
    ----------
    checkpoint_path: path to checkpoint
    average_treatment_effect_method: method to compute average treatment effect. "perturbseq"
        normalizes for library size and applies log transform before assessing effect
    batch_size: batch size to use for IWELBO computation

    Returns
    -------
    dictionary with test set metrics
    """
    lightning_module = load_checkpoint(checkpoint_path)
    data_module = lightning_module.get_data_module()
    predictor = lightning_module.predictor

    metrics = {}

    # compute test set IWELBO
    test_loader = DataLoader(
        data_module.test_dataloader().dataset,
        batch_size=batch_size,
    )
    test_iwelbo_df = predictor.compute_predictive_iwelbo(
        loaders=test_loader, n_particles=100
    )
    test_iwelbo = test_iwelbo_df["IWELBO"].mean()
    metrics["test/IWELBO"] = test_iwelbo

    # assess correlation between estimated average treatment effects from model and data
    data_ate = data_module.get_estimated_average_treatment_effects(
        method=average_treatment_effect_method
    )

    if data_ate is not None:
        model_ate = predictor.estimate_average_effects_data_module(
            data_module=data_module,
            control_label=data_ate.uns["control"],
            method=average_treatment_effect_method,
            n_particles=ate_n_particles,
            condition_values=dict(library_size=10000 * torch.ones((1,))),
            batch_size=batch_size,
        )

        data_ate, model_ate = align_adatas(data_ate, model_ate)

        intervention_info = data_module.get_unique_observed_intervention_info()

        metrics["ATE_n_particles"] = ate_n_particles

        # compute average treatment effect metrics for all perturbations
        ate_metrics_all_splits = get_ate_metrics(data_ate, model_ate)
        for k, v in ate_metrics_all_splits.items():
            metrics[f"{k}-all"] = v

        # compute average treatment effect metrics for perturbations available
        # in each split
        for split in ["train", "val", "test"]:
            split_perturbations = intervention_info[intervention_info[split]].index
            idx = data_ate.obs.index.isin(split_perturbations)
            ate_metrics_split = get_ate_metrics(data_ate[idx], model_ate[idx])
            for k, v in ate_metrics_split.items():
                metrics[f"{k}-{split}"] = v

    return metrics


def get_ate_metrics(data_ate, model_ate):
    metrics = {}
    top_20_idx = np.argpartition(np.abs(data_ate.X.copy()), data_ate.shape[1] - 20)[
        :, -20:
    ]

    x = data_ate.X.flatten()
    y = model_ate.X.flatten()

    metrics["ATE_pearsonr"] = pearsonr(x, y)[0]
    metrics["ATE_r2"] = r2_score(x, y)

    # evaluate correlation / R2 across top 20 DE genes per perturbation
    x = np.take_along_axis(data_ate.X.copy(), top_20_idx, axis=-1).flatten()
    y = np.take_along_axis(model_ate.X.copy(), top_20_idx, axis=-1).flatten()

    metrics["ATE_pearsonr_top20"] = pearsonr(x, y)[0]
    metrics["ATE_r2_top20"] = r2_score(x, y)

    return metrics


def evaluate_local_experiment(
    experiment_path: str,
    average_treatment_effect_method: Literal["mean", "perturbseq"],
    batch_size: int = 128,
    ate_n_particles: int = 2500,
):
    """
    Compute and save evaluation metrics for checkpoint with best eval loss in
     local experiment to `{experiment_path}/test_metrics.csv`

    Parameters
    ----------
    experiment_path: path to experiment (typically in results/ directory)
    average_treatment_effect_method
    batch_size: batch size used during IWELBO computation
    """
    checkpoint_names = os.listdir(join(experiment_path, "checkpoints"))
    # TODO: add better logic if needed
    best_checkpoints = [x for x in checkpoint_names if x[:4] == "best"]
    assert len(best_checkpoints) == 1
    checkpoint_path = join(experiment_path, "checkpoints", best_checkpoints[0])
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    metrics = evaluate_checkpoint(
        checkpoint_path,
        average_treatment_effect_method=average_treatment_effect_method,
        batch_size=batch_size,
        ate_n_particles=ate_n_particles,
    )
    metrics["checkpoint"] = checkpoint_name

    metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()}).T
    metrics_path = join(experiment_path, "test_metrics.csv")
    metrics_df.to_csv(metrics_path)


def evaluate_local_checkpoint(
    checkpoint_path: str,
    average_treatment_effect_method: Literal["mean", "perturbseq"],
    batch_size: int = 128,
    ate_n_particles: int = 2500,
):
    """
    Compute and save evaluation metrics specified checkpoint_path,
    saves results to {checkpoint_path}_test_metrics.csv

    Parameters
    ----------
    experiment_path: path to experiment (typically in results/ directory)
    average_treatment_effect_method
    batch_size: batch size used during IWELBO computation
    """
    checkpoint_base = splitext(checkpoint_path)[0]
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    metrics = evaluate_checkpoint(
        checkpoint_path,
        average_treatment_effect_method=average_treatment_effect_method,
        batch_size=batch_size,
        ate_n_particles=ate_n_particles,
    )
    metrics["checkpoint"] = checkpoint_name

    metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()}).T
    metrics_path = checkpoint_base + "_test_metrics.csv"
    metrics_df.to_csv(metrics_path)


def evaluate_wandb_experiment(
    experiment_path: str,
    average_treatment_effect_method: Literal["mean", "perturbseq"],
    batch_size: int = 128,
    ate_n_particles: int = 2500,
):
    """
    Compute and save evaluation metrics for checkpoint with best eval loss
    Metrics are saved to wandb run summary
    """
    api = wandb.Api()
    run = api.run(experiment_path)

    # TODO: improve logic if needed
    run_file_paths = [x.name for x in run.files()]
    best_checkpoint_paths = [
        x
        for x in run_file_paths
        if os.path.split(x)[0] == "checkpoints" and "best" in x
    ]
    assert len(best_checkpoint_paths) == 1
    wandb_file = run.file(best_checkpoint_paths[0])

    # download checkpoint
    basedir = run.name + "/"
    os.makedirs(basedir, exist_ok=True)
    checkpoint_path = wandb_file.download(root=basedir, replace=True).name

    metrics = evaluate_checkpoint(
        checkpoint_path,
        average_treatment_effect_method=average_treatment_effect_method,
        batch_size=batch_size,
        ate_n_particles=ate_n_particles,
    )

    # save metrics to run summary
    for k in metrics:
        run.summary[k] = metrics[k]

    run.summary.update()


def load_checkpoint(checkpoint_path: str):
    lightning_module = TrainConfigPerturbationLightningModule.load_from_checkpoint(
        checkpoint_path
    )
    return lightning_module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--perturbseq", action="store_true")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--ate_n_particles", type=int, default=2500)

    args = parser.parse_args()

    method: Literal["mean", "perturbseq"] = "perturbseq" if args.perturbseq else "mean"
    if args.wandb:
        evaluate_wandb_experiment(
            args.experiment_path,
            method,
            batch_size=args.batch_size,
            ate_n_particles=args.ate_n_particles,
        )
    elif os.path.isdir(args.experiment_path):
        evaluate_local_experiment(
            args.experiment_path,
            method,
            batch_size=args.batch_size,
            ate_n_particles=args.ate_n_particles,
        )
    else:
        evaluate_local_checkpoint(
            args.experiment_path,
            method,
            batch_size=args.batch_size,
            ate_n_particles=args.ate_n_particles,
        )
