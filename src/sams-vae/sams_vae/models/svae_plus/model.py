from typing import Dict, List, Optional, Tuple

import torch
from torch.distributions import Bernoulli, Beta, Distribution, Normal

from sams_vae.models.utils.mlp import LIKELIHOOD_KEY_DTYPE, get_likelihood_mlp


class SVAEPlusModel(torch.nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_treatments: int,
        n_phenos: int,
        mask_beta_concentration_1: float,
        mask_beta_concentration_2: float,
        likelihood_key: LIKELIHOOD_KEY_DTYPE,
        decoder_n_layers: int,
        decoder_n_hidden: int,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_treatments = n_treatments
        self.n_phenos = n_phenos
        self.likelihood_key = likelihood_key
        self.decoder_n_layers = decoder_n_layers
        self.decoder_n_hidden = decoder_n_hidden

        # mask parameters
        self.register_buffer(
            "mask_beta_concentration_1",
            mask_beta_concentration_1 * torch.ones((n_treatments, n_latent)),
        )
        self.register_buffer(
            "mask_beta_concentration_2",
            mask_beta_concentration_2 * torch.ones((n_treatments, n_latent)),
        )

        # learned prior mean for p(z|a)
        self.action_prior_mean = torch.nn.Parameter(
            0.1 * torch.randn((n_treatments, n_latent))
        )

        self.decoder = get_likelihood_mlp(
            likelihood_key=likelihood_key,
            n_input=n_latent,
            n_output=n_phenos,
            n_layers=decoder_n_layers,
            n_hidden=decoder_n_hidden,
            use_batch_norm=False,
            activation_fn=torch.nn.LeakyReLU,
        )

    def get_var_keys(self) -> List[str]:
        return ["z", "mask_prob", "mask"]

    def forward(
        self,
        D: torch.Tensor,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
    ) -> Tuple[Dict[str, Distribution], Dict[str, torch.Tensor]]:
        """
        Sample from generative process, conditioned on D and other condition_values

        Parameters
        ----------
        D: dosages (n, n_perturbations)
        condition_values: optional dictionary of conditioning values (missing variables
                are sampled from prior)
        n_particles: number of samples to draw from generative distribution
                for each observed dosage

        Returns
        -------
        Tuple with dictionaries of generative distributions and samples, each
        with batch size n_particles. Keys are strings specifying variables
        """
        if condition_values is None:
            condition_values = dict()

        # define generative distribution and samples
        generative_dists = {}
        samples = {}

        generative_dists["p_mask_prob"] = Beta(
            self.mask_beta_concentration_1,
            self.mask_beta_concentration_2,
        )
        if "mask_prob" not in condition_values:
            samples["mask_prob"] = generative_dists["p_mask_prob"].sample(
                (n_particles,)
            )
        else:
            samples["mask_prob"] = condition_values["mask_prob"]

        generative_dists["p_mask"] = Bernoulli(samples["mask_prob"])
        if "mask" not in condition_values:
            # already have n_particles in dist parameters from mask_prob
            samples["mask"] = generative_dists["p_mask"].sample()
        else:
            samples["mask"] = condition_values["mask"]

        assert (D != 0).sum(-1).max() <= 1, "SVAE+ expects one-hot perturbation dosages"

        perturbation_idx = (D != 0).float().argmax(dim=-1)
        z_prior_mean = quick_select(
            self.action_prior_mean, dim=-2, index=perturbation_idx
        )
        z_prior_mask = quick_select(samples["mask"], dim=-2, index=perturbation_idx)

        # set z prior mean to be 0 if all dosages are 0
        # in previous step, argmax will assign all 0's to index 0
        # ok to set just mean bc multiplied with mask
        z_prior_mean[(D != 0).sum(-1) == 0] = 0

        generative_dists["p_z"] = Normal(
            z_prior_mask * z_prior_mean, torch.ones_like(z_prior_mean)
        )
        if "z" not in condition_values:
            samples["z"] = generative_dists["p_z"].rsample()
        else:
            samples["z"] = condition_values["z"]

        if self.likelihood_key != "library_nb":
            generative_dists["p_x"] = self.decoder(samples["z"])
        else:
            generative_dists["p_x"] = self.decoder(
                samples["z"], condition_values["library_size"]
            )

        samples["x"] = generative_dists["p_x"].sample()

        return generative_dists, samples


def quick_select(tensor, dim, index):
    # returns a new tensor selected by index along specified dim
    # tensor does not share storage (due to index_select)
    if dim != 0:
        tensor = torch.transpose(tensor, 0, dim)
    tensor = torch.index_select(tensor, dim=0, index=index)
    if dim != 0:
        tensor = torch.transpose(tensor, 0, dim)
    return tensor
