from typing import Any, Dict, List, Literal, Optional

import anndata
import numpy as np
import pandas as pd
import scipy as sp
import torch
from tqdm.auto import tqdm


def estimate_data_average_treatment_effects(
    adata: anndata.AnnData,
    label_col: str,
    control_label: Any,
    method: Literal["mean", "perturbseq"],
    compute_fdr: bool = False,
) -> anndata.AnnData:
    """

    Parameters
    ----------
    adata: AnnData containing observations and annotated perturbations. Observations
            should be in X
    label_col: column in adata obs dataframe with labels of unique perturbations to
            compute average treatment effects
    control_label: value in label_col to compute effects relative to
    method: method key for computing average treatment effect. "mean" is standard average
            effect, "perturbseq" is effect after normalizing for library size and applying log
    compute_fdr: whether to compute false discovery rate


    Returns
    -------
    AnnData with average treatment effects in X, obs index with perturbation annotations,
    and control perturbation in uns
    """
    if compute_fdr:
        raise NotImplementedError

    valid_methods = ["mean", "perturbseq"]
    assert method in valid_methods, f"Method must be one of {valid_methods}"

    perturbations = adata.obs[label_col].unique()
    assert control_label in perturbations
    alt_labels = [x for x in perturbations if x != control_label]

    X_control = adata[adata.obs[label_col] == control_label].X
    if sp.sparse.issparse(X_control):
        X_control = X_control.toarray()

    if method == "perturbseq":
        X_control = 1e4 * X_control / np.sum(X_control, axis=1, keepdims=True)
        X_control = np.log2(X_control + 1)
    X_control_mean = X_control.mean(0)

    average_effects = []

    for alt_label in tqdm(alt_labels):
        X_alt = adata[adata.obs[label_col] == alt_label].X
        if sp.sparse.issparse(X_alt):
            X_alt = X_alt.toarray()
        if method == "perturbseq":
            X_alt = 1e4 * X_alt / np.sum(X_alt, axis=1, keepdims=True)
            X_alt = np.log2(X_alt + 1)
        X_alt_mean = X_alt.mean(0)
        average_effects.append(X_alt_mean - X_control_mean)

    average_effects = np.stack(average_effects)
    results = anndata.AnnData(
        obs=pd.DataFrame(index=alt_labels),
        X=average_effects,
        var=adata.var.copy(),
        uns=dict(control=control_label),
    )
    return results


@torch.no_grad()
def estimate_model_average_treatment_effect(
    model: torch.nn.Module,
    guide: torch.nn.Module,
    dosages_alt: torch.Tensor,
    dosages_control: torch.Tensor,
    n_particles: int,
    method: Literal["mean", "perturbseq"],
    condition_values: Optional[Dict[str, torch.Tensor]] = None,
    batch_size: int = 500,
    dosage_independent_variables: Optional[List[str]] = None,
    seed: int = 0,
):
    torch.manual_seed(seed)

    valid_methods = ["mean", "perturbseq"]
    assert method in valid_methods, f"Method must be one of {valid_methods}"

    if condition_values is None:
        condition_values = dict()

    # get device with model parameters (assumes model on a single device)
    device = next(model.parameters()).device

    # preprocess extra conditioning values
    for k, v in condition_values.items():
        # match shape to number of particles in a batch
        if len(v.shape) == 2:
            condition_values[k] = v.unsqueeze(0).expand((batch_size, -1, -1))

        # move to device with model parameters
        condition_values[k] = v.to(device)

    # get n_phenos
    curr_condition_values = {k: v[:1] for k, v in condition_values.items()}
    _, model_samples = model(
        D=dosages_control, condition_values=curr_condition_values, n_particles=1
    )
    n_phenos = model_samples["x"].shape[-1]

    # compute estimated average value of features under control and alternate
    # conditions
    # performs scaling and log transform if method == "perturbseq"

    X_control_sum = torch.zeros((1, n_phenos), device=device)
    X_alt_sums = torch.zeros((dosages_alt.shape[0], n_phenos), device=device)

    for i in range(0, n_particles, batch_size):
        # compute with up to batch_size particles at once
        curr_num_particles = min(batch_size, n_particles - i)

        curr_condition_values = {}

        # add passed in condition values
        for k, v in condition_values.items():
            if v.shape[0] == n_particles:
                curr_condition_values[k] = v[i : i + curr_num_particles].to(device)
            else:
                curr_condition_values[k] = v.to(device)

        # sample parameters from guide
        guide_dists, guide_samples = guide(
            n_particles=curr_num_particles, condition_values=curr_condition_values
        )

        for k, v in guide_samples.items():
            curr_condition_values[k] = v

        # sample under control perturbation using model
        _, model_samples = model(
            D=dosages_control,
            condition_values=curr_condition_values,
            n_particles=curr_num_particles,
        )

        # hold latent variables that do not depend on treatment constant
        if dosage_independent_variables is not None:
            for k in dosage_independent_variables:
                curr_condition_values[k] = model_samples[k]

        # shape: (n_particles, 1, n_phenos)
        X_control = model_samples["x"].squeeze(1)
        if method == "perturbseq":
            # standardize by library size and log normalize
            X_control = 1e4 * X_control / torch.sum(X_control, dim=1, keepdim=True)
            X_control = torch.log2(X_control + 1)
        X_control_sum[0] += X_control.sum(0)

        for t_idx in tqdm(range(dosages_alt.shape[0])):
            D_curr = dosages_alt[t_idx : t_idx + 1]

            # sample under alternate perturbation using model
            _, model_samples = model(
                D=D_curr,
                condition_values=curr_condition_values,
                n_particles=curr_num_particles,
            )

            # shape: (n_particles, 1, n_phenos)
            X_curr = model_samples["x"].squeeze(1)
            if method == "perturbseq":
                # standardize by library size and log normalize
                X_curr = 1e4 * X_curr / torch.sum(X_curr, dim=1, keepdim=True)
                X_curr = torch.log2(X_curr + 1)
            X_alt_sums[t_idx] += X_curr.sum(0)

    # compute estimated average treatment effect from
    # estimated average means under each condition
    X_control_mean = X_control_sum / n_particles
    X_alt_means = X_alt_sums / n_particles
    average_effects = (X_alt_means - X_control_mean).detach().cpu().numpy()

    return average_effects
