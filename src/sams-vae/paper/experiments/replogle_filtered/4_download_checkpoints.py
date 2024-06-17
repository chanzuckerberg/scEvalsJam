import argparse
import os

import pandas as pd

import wandb

N_LATENT = 100

CHECKPOINT_DIRECTORY = "results/checkpoints/"
PRECOMPUTED_CHECKPOINT_URL = "s3://insitro-research-2023-sams-vae/replogle/checkpoints/"


def download_precomputed_checkpoints():
    os.makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)
    os.system(
        f"aws s3 cp --recursive {PRECOMPUTED_CHECKPOINT_URL} {CHECKPOINT_DIRECTORY}"
    )


def download_best_checkpoints():
    os.makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)

    combined_results = pd.read_csv("results/replogle_results.csv")

    # load best checkpoint for each model (based on validation loss)
    best_settings_df = (
        combined_results[combined_results["n_latent"] == N_LATENT]
        .groupby(
            [
                "n_latent",
                "model",
                "guide",
                "mask_prior_prob",
                "mask_beta_concentration_2",
                "mean_field_encoder",
            ],
            dropna=False,
        )[["val/IWELBO", "test/IWELBO", "ATE_pearsonr-all"]]
        .agg(["mean", "std", "count"])
        .sort_values(("val/IWELBO", "mean"), ascending=False)
        .reset_index()
        .drop_duplicates(
            [
                ("model", ""),
                ("guide", ""),
                ("mask_prior_prob", ""),
                ("mean_field_encoder", ""),
            ]
        )
    )

    for i in range(best_settings_df.shape[0]):
        row = best_settings_df.iloc[i]
        idx = combined_results["n_latent"] == N_LATENT
        idx = idx & (combined_results["model"] == row["model"].item())
        idx = idx & (combined_results["guide"] == row["guide"].item())
        if not pd.isna(row["mean_field_encoder"].item()):
            idx = idx & (
                combined_results["mean_field_encoder"]
                == row["mean_field_encoder"].item()
            )
        if not pd.isna(row["mask_prior_prob"].item()):
            idx = idx & (
                combined_results["mask_prior_prob"] == row["mask_prior_prob"].item()
            )
        if not pd.isna(row["mask_beta_concentration_2"].item()):
            idx = idx & (
                combined_results["mask_beta_concentration_2"]
                == row["mask_beta_concentration_2"].item()
            )
        curr = combined_results[idx]
        # select checkpoint with best test IWELBO from best hyperparameter setting
        # (selected with validation loss)
        best_run_path = curr.sort_values("test/IWELBO", ascending=False)["path"].iloc[0]
        print(download_checkpoints(best_run_path))


def download_checkpoints(run_path):
    api = wandb.Api()
    run = api.run(run_path)
    checkpoint_wandb_files = [
        x for x in run.files() if os.path.split(x.name)[0] == "checkpoints"
    ]

    basedir = os.path.join("results/checkpoints/", run.name)
    os.makedirs(basedir, exist_ok=True)

    checkpoint_paths = []
    for wandb_file in checkpoint_wandb_files:
        checkpoint_path = wandb_file.download(root=basedir, replace=True).name
        checkpoint_paths.append(checkpoint_path)
    return checkpoint_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precomputed", action="store_true")
    args = parser.parse_args()

    if args.precomputed:
        download_precomputed_checkpoints()
    else:
        download_best_checkpoints()
