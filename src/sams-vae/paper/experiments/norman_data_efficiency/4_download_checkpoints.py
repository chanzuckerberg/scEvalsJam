import argparse
import os

import pandas as pd

import wandb

N_LATENT = 100

CHECKPOINT_DIRECTORY = "results/checkpoints/"
PRECOMPUTED_CHECKPOINT_URL = (
    "s3://insitro-research-2023-sams-vae/norman_data_efficiency/checkpoints/"
)


def download_precomputed_checkpoints():
    os.makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)
    os.system(
        f"aws s3 cp --recursive {PRECOMPUTED_CHECKPOINT_URL} {CHECKPOINT_DIRECTORY}"
    )


def download_wandb_checkpoints():
    os.makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)

    results = pd.read_csv("results/norman_data_efficiency_results.csv")

    # download model checkpoints trained on 100% of combinations
    df = results.copy()
    df = df[
        df["model"].isin(
            [
                "sams_vae_correlated",
                "sams_vae_mean_field",
                "svae+",
                "cpa_vae",
                "conditional_vae",
            ]
        )
    ]
    df = df[df["n_layers"] == 1]
    df = df[(df["encode_unique"] == False) | (df["model"] == "svae+")]  # noqa: E712
    df = df[
        (df["mean_field_encoder"] == False)  # noqa: E712
        | pd.isna(df["mean_field_encoder"])
    ]
    df = df[df["data_module_kwargs.frac_combination_cells_train"] == 1]
    df = df[df["split_seed"] == 0]
    paths = df.sort_values("val/IWELBO", ascending=False).drop_duplicates("model")[
        "path"
    ]

    for path in paths:
        download_checkpoints(path)


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
        download_wandb_checkpoints()
