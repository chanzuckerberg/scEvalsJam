from typing import Optional

import anndata
import numpy as np
import pandas as pd
import torch
from torch.distributions import Bernoulli, Normal


def simulate_data_sams_vae(
    n_latent: int = 15,
    n_treatments: int = 100,
    n_phenos: int = 100,
    n_hidden_layers_decoder: int = 4,
    n_hidden_dim_decoder: int = 50,
    decoder_exp_output: bool = False,
    decoder_init_strategy: str = "orthogonal",
    decoder_layer_sparsity: Optional[float] = None,
    mask_prior_prob: float = 2 / 15,
    latent_effect_magnitude_mean: float = 5,
    latent_effect_magnitude_std: float = 0.5,
    n_samples_per_treatment_train: int = 500,
    n_samples_per_treatment_val: int = 100,
    n_samples_per_treatment_test: int = 100,
    frac_var_pheno_noise: float = 0.2,
    seed: int = 0,
    return_decoder: bool = False,
):
    """
    Simulate data with known latent effects.

    Generative process:
    mask ~ Bernoulli(mask_prior_prob)
    E_{sign} ~ Bernoulli(0.5)
    E_{magnitude} ~ N(latent_effect_magnitude_mean, latent_effect_magnitude_std)
    E = (E_{sign} * 2 - 1) * E_{magnitude}
    z_{basal} ~ N(0, 1)
    z = z_{basal} + (mask * E) d

    mean = decoder(z)
    mean = mean / mean.std(0)
    if decoder_exp_output:
        mean = exp(mean)

    # pheno_scale selected for specified value of norman frac var pheno noise
    x ~ Norman(mean, pheno_scale)

    Key aspects:
    1. Embedding is not sampled from N(0, 1)--sampled to allow specification of
       mean / variance in magnitude
    2. Can specify to apply exponential to output of decoder, which will result
       in larger outliers due to treatments.
       TODO: better control outliers (distribution of phenotype
       fraction variance due to treatment?)
    3. Can specify fraction of variance due to phenotype noise

    TODO: add more distributions?

    Parameters
    ----------
    n_latent
    n_treatments
    n_phenos
    n_hidden_decoder
    decoder_exp_output
    mask_prior_prob
    latent_effect_magnitude_mean
    latent_effect_magnitude_std
    n_samples_per_treatment_train
    n_samples_per_treatment_val
    n_samples_per_treatment_test
    frac_var_pheno_noise
    seed

    Returns
    -------

    """
    # set random seed
    torch.manual_seed(seed)

    # simulate latent effects
    mask = Bernoulli(mask_prior_prob).sample((n_treatments, n_latent))
    embedding = Normal(
        latent_effect_magnitude_mean, latent_effect_magnitude_std
    ).sample((n_treatments, n_latent))
    sign = (2 * Bernoulli(0.5).sample((n_treatments, n_latent))) - 1
    embedding = sign * embedding
    latent_effects = mask * embedding

    # initialize phenotype decoder
    decoder = get_decoder(
        n_latent,
        n_phenos,
        n_hidden_layers_decoder,
        n_hidden_dim_decoder,
        init_strategy=decoder_init_strategy,
        sparsity=decoder_layer_sparsity,
    )

    # sample phenotype
    total_samples = (
        n_samples_per_treatment_train
        + n_samples_per_treatment_val
        + n_samples_per_treatment_test
    )
    D = torch.eye(n_treatments).repeat((total_samples, 1))
    z_basal = Normal(0, 1).sample((D.shape[0], n_latent))
    with torch.no_grad():
        z = z_basal + torch.matmul(D, latent_effects)
        pheno_mean = decoder(z)
        pheno_mean_std = pheno_mean.std(0)
        pheno_mean_std[pheno_mean_std == 0] = 1
        pheno_mean = pheno_mean / pheno_mean_std

    if decoder_exp_output:
        pheno_mean = pheno_mean.exp()

    pheno_obs_noise_var = (frac_var_pheno_noise * pheno_mean.var(0)) / (
        1 - frac_var_pheno_noise
    )
    pheno_obs_noise_var[pheno_obs_noise_var == 0] = 1
    x = Normal(pheno_mean, pheno_obs_noise_var.sqrt()).sample()

    split = []
    split += ["train"] * (n_samples_per_treatment_train * n_treatments)
    split += ["val"] * (n_samples_per_treatment_val * n_treatments)
    split += ["test"] * (n_samples_per_treatment_test * n_treatments)

    obs = pd.DataFrame({"treatment": D.argmax(1), "split": split})
    obs["treatment"] = obs["treatment"].astype(str)
    d_var_info = pd.DataFrame({"treatment": [str(i) for i in range(n_treatments)]})
    d_var_info = d_var_info.set_index("treatment")
    layers = dict(mean=pheno_mean.numpy())
    obsm = dict(z_basal=z_basal.numpy(), D=D.numpy(), z=z)
    uns = dict(
        latent_effects=latent_effects.numpy(),
        D_unique=np.eye(n_treatments),
        D_var_info=d_var_info,
    )
    adata = anndata.AnnData(X=x.numpy(), layers=layers, obs=obs, obsm=obsm, uns=uns)
    ret = adata if not return_decoder else (adata, decoder)
    return ret


def get_decoder(
    n_latent,
    n_phenos,
    n_hidden_layers_decoder,
    n_hidden_dim_decoder,
    init_strategy="orthogonal",
    sparsity=None,
):
    decoder_layers = []
    curr_dim = n_latent
    for i in range(n_hidden_layers_decoder):
        layer = torch.nn.Linear(curr_dim, n_hidden_dim_decoder, bias=False)
        init_decoder_layer(layer, sparsity=sparsity, init_strategy=init_strategy)
        activation = torch.nn.LeakyReLU(negative_slope=0.2)
        curr_dim = n_hidden_dim_decoder
        decoder_layers += [layer, activation]
    layer = torch.nn.Linear(curr_dim, n_phenos, bias=False)
    init_decoder_layer(layer, sparsity=sparsity, init_strategy=init_strategy)
    decoder_layers.append(layer)
    decoder = torch.nn.Sequential(*decoder_layers)
    return decoder


def init_decoder_layer(layer, sparsity=None, init_strategy="orthogonal"):
    if init_strategy == "orthogonal":
        torch.nn.init.orthogonal_(layer.weight)
    elif init_strategy == "normal":
        torch.nn.init.kaiming_normal_(layer.weight)
    else:
        torch.nn.init.sparse_(layer.weight.T, sparsity, 1)
