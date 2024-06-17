# Code adapted from Lopez et al (https://arxiv.org/abs/2211.03553)

from typing import Optional

import numpy as np
import pandas as pd
import torch
from anndata import AnnData


def simulate_svae_plus_paper_dataset(
    n_cells_per_chem_tr: int = 500,
    n_cells_per_chem_val: int = 100,
    n_cells_per_chem_test: int = 100,
    n_chem: int = 100,
    n_latent: int = 15,
    n_genes: int = 100,
    mean_shift_size: float = 5.0,
    bernoulli_mask_prob: Optional[float] = None,
) -> AnnData:
    """
    Simulated data from SVAE+ paper (https://arxiv.org/abs/2211.03553)
    Default arguments set to match paper, though main text and methods contradict:
    Main text: "In all cases, we simulated data subject to K = 100 interventions,
    with 500 cells sampled per intervention. Measurements from 20 interventions
    were held-out from the training set, and used for transfer learning evaluation."
    Methods: "We use K = 100 interventions in our simulations, and sample N = 100,000
    cells in total. The first 80 interventions form the training set, and the last 20
    form the test set. Each intervention a ∈ [K] is represented by a sparse perturbation
    embedding µa ∈ R p, where p = 15 is the dimension of the embedding."
    Not sure what was used for n_genes? Will keep default from code

    Added option to sample mask from bernoulli
    """
    np.random.seed(0)
    torch.manual_seed(0)

    # start by generating target for every chemical
    if bernoulli_mask_prob is None:
        targets = np.zeros((n_chem, n_latent))
        for chem in range(n_chem):
            num_target = np.random.choice([1, 2, 3])
            target_index = np.random.choice(np.arange(n_latent), num_target)
            targets[chem, target_index] += 1
    else:
        targets = (
            torch.distributions.Bernoulli(bernoulli_mask_prob)
            .sample((n_chem, n_latent))
            .numpy()
        )

    # now generate shifts and apply mask to get target effects
    shift_sign = (
        2 * (np.random.uniform(0, 1, size=targets.shape) > 0.5).astype(float) - 1
    )
    shift_abs = np.random.normal(mean_shift_size, 0.5, size=targets.shape)
    action_specific_prior_mean = targets * shift_abs * shift_sign

    # now we can go around simulating cells and z for all those cells
    n_cells_per_chem = (
        n_cells_per_chem_tr + n_cells_per_chem_val + n_cells_per_chem_test
    )
    z = np.zeros((n_chem * n_cells_per_chem, n_latent))
    z0 = np.zeros((n_chem * n_cells_per_chem, n_latent))
    split_labels = []
    for chem in range(n_chem):
        z[
            chem * n_cells_per_chem : (chem + 1) * n_cells_per_chem
        ] = np.random.multivariate_normal(
            action_specific_prior_mean[chem], np.eye(n_latent), size=(n_cells_per_chem)
        )
        z0[chem * n_cells_per_chem : (chem + 1) * n_cells_per_chem] = (
            z[chem * n_cells_per_chem : (chem + 1) * n_cells_per_chem]
            - action_specific_prior_mean[chem]
        )
        split_labels += ["train"] * n_cells_per_chem_tr
        split_labels += ["val"] * n_cells_per_chem_val
        split_labels += ["test"] * n_cells_per_chem_test

    # finally, get the decoder, and get gene expression x for the cells
    params = _prepare_params_decoder(n_genes, n_latent)
    x = _decoder(z, params=params)
    x = np.random.poisson(lam=1e6 * x)
    # put label in there
    y = np.concatenate([n_cells_per_chem * [chem] for chem in range(n_chem)])

    # shuffle dataset
    # ind = np.random.permutation(np.arange(n_cells_per_chem * n_chem))
    # x = x[ind]
    # y = y[ind]
    # z = z[ind]

    # dump into anndata
    adata = AnnData(x, dtype=np.float32)
    adata.obs["T"] = pd.Categorical(y.astype(str))
    adata.obs["split"] = split_labels
    adata.obsm["groundtruth_latent"] = z
    adata.obsm["groundtruth_z0"] = z0
    adata.uns["prior_mean"] = action_specific_prior_mean
    adata.uns["prior_mean_idx"] = np.arange(action_specific_prior_mean.shape[0]).astype(
        str
    )
    return adata


# code adapted from lachapelle et al. (their code assumes x_dim = z_dim = h_dim)
def _prepare_params_decoder(x_dim, z_dim, h_dim=40, neg_slope=0.2):
    if z_dim > h_dim or h_dim > x_dim:
        raise ValueError("CHECK dim <= h_dim <= x_dim")
    # sampling NN weight matrices
    W1 = np.random.normal(size=(z_dim, h_dim))
    W1 = np.linalg.qr(W1.T)[0].T
    W1 *= np.sqrt(2 / (1 + neg_slope**2)) * np.sqrt(2.0 / (z_dim + h_dim))

    W2 = np.random.normal(size=(h_dim, h_dim))
    W2 = np.linalg.qr(W2.T)[0].T
    # print("distance to identity:", np.max(np.abs(np.matmul(W2, W2.T) - np.eye(h_dim)))
    W2 *= np.sqrt(2 / (1 + neg_slope**2)) * np.sqrt(2.0 / (2 * h_dim))

    W3 = np.random.normal(size=(h_dim, h_dim))
    W3 = np.linalg.qr(W3.T)[0].T
    # print("distance to identity:", np.max(np.abs(np.matmul(W3, W3.T) - np.eye(h_dim))))
    W3 *= np.sqrt(2 / (1 + neg_slope**2)) * np.sqrt(2.0 / (2 * h_dim))

    W4 = np.random.normal(size=(h_dim, x_dim))
    W4 = np.linalg.qr(W4.T)[0].T
    # print("distance to identity:", np.max(np.abs(np.matmul(W4, W4.T) - np.eye(h_dim))))
    W4 *= np.sqrt(2 / (1 + neg_slope**2)) * np.sqrt(2.0 / (x_dim + h_dim))
    return {"W1": W1, "W2": W2, "W3": W3, "W4": W4}


def _decoder(z, params, neg_slope=0.2):
    W1, W2, W3, W4 = params["W1"], params["W2"], params["W3"], params["W4"]
    # note that this decoder is almost surely invertible WHEN dim <= h_dim <= x_dim
    # since Wx is injective
    # when columns are linearly indep, which happens almost surely,
    # plus, composition of injective functions is injective.
    h1 = np.matmul(z, W1)
    h1 = np.maximum(neg_slope * h1, h1)  # leaky relu
    h2 = np.matmul(h1, W2)
    h2 = np.maximum(neg_slope * h2, h2)  # leaky relu
    h3 = np.matmul(h2, W3)
    h3 = np.maximum(neg_slope * h3, h3)  # leaky relu
    logits = np.matmul(h3, W4)
    logits /= np.std(logits)
    e_x = np.exp(logits - np.max(logits))
    return e_x / e_x.sum(axis=0)
