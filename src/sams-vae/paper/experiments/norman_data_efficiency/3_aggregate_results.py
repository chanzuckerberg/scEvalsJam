import argparse
import os

import pandas as pd
import yaml

import wandb

RESULTS_CSV_PATH = "results/norman_data_efficiency_results.csv"
PRECOMPUTED_URL = "s3://insitro-research-2023-sams-vae/norman_data_efficiency/norman_data_efficiency_results.csv"  # noqa: E501


def download_precomputed_results():
    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    os.system(f"aws s3 cp {PRECOMPUTED_URL} {RESULTS_CSV_PATH}")


def aggregate_results():
    with open("sweep_ids.yaml") as f:
        sweep_ids = yaml.safe_load(f)

    results = {}
    for k, v in sweep_ids.items():
        results[k] = load_results(v)

    combined_results = []
    for k, df in results.items():
        df["model"] = k
        combined_results.append(df)
    combined_results = pd.concat(combined_results)

    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    combined_results.to_csv(RESULTS_CSV_PATH, index=False)


def load_results(sweep_id):
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    stats_list = []
    for run in sweep.runs:
        curr = {}
        curr["name"] = run.name
        curr["path"] = "/".join(run.path)
        for k, v in run.config.items():
            curr[k] = v
            # simplify some of the key names
            if "decoder_n_layers" in k:
                curr["n_layers"] = v
            if "n_latent" in k:
                curr["n_latent"] = v
            if k == "lightning_module_kwargs.lr":
                curr["lr"] = v
            if k == "model_kwargs.mask_prior_prob":
                curr["mask_prior_prob"] = v
            if k == "model_kwargs.mask_beta_concentration_2":
                curr["mask_beta_concentration_2"] = v
            if k == "guide_kwargs.mean_field_encoder":
                curr["mean_field_encoder"] = v
            if k == "data_module_kwargs.frac_combination_cells_train":
                curr["frac_combinations_train"] = v
            if k == "data_module_kwargs.encode_combos_as_unique":
                curr["encode_unique"] = v
            if k == "data_module_kwargs.split_seed":
                curr["split_seed"] = v
        for k, v in run.summary.items():
            curr[k] = v
        stats_list.append(pd.Series(curr))
    stats = pd.concat(stats_list, axis=1).T.set_index("name")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precomputed", action="store_true")
    args = parser.parse_args()

    if args.precomputed:
        download_precomputed_results()
    else:
        aggregate_results()
