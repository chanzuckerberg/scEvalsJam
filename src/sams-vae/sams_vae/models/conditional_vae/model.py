from typing import Dict, List, Optional, Tuple

import torch
from torch.distributions import Distribution, Normal

from sams_vae.models.utils.mlp import LIKELIHOOD_KEY_DTYPE, get_likelihood_mlp


class ConditionalVAEModel(torch.nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_treatments: int,
        n_phenos: int,
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

        self.decoder = get_likelihood_mlp(
            likelihood_key=likelihood_key,
            n_input=n_latent + n_treatments,
            n_output=n_phenos,
            n_layers=decoder_n_layers,
            n_hidden=decoder_n_hidden,
            use_batch_norm=False,
            activation_fn=torch.nn.LeakyReLU,
        )

    def get_var_keys(self) -> List[str]:
        return ["z"]

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
        n = D.shape[0]
        device = D.device

        if condition_values is None:
            condition_values = dict()

        # define generative distribution and samples
        generative_dists = {}
        samples = {}

        generative_dists["p_z"] = Normal(
            torch.zeros((n, self.n_latent)).to(device),
            torch.ones((n, self.n_latent)).to(device),
        )

        if "z" not in condition_values:
            samples["z"] = generative_dists["p_z"].sample((n_particles,))
        else:
            samples["z"] = condition_values["z"]

        # concatenate treatment dosages to z
        D_reshaped = D.unsqueeze(0).expand(n_particles, -1, -1)  # expand to n_particles
        # n_particles x n x (n_latent + n_treatments)
        decoder_input = torch.cat([samples["z"], D_reshaped], dim=-1)

        if self.likelihood_key != "library_nb":
            generative_dists["p_x"] = self.decoder(decoder_input)
        else:
            generative_dists["p_x"] = self.decoder(
                decoder_input, condition_values["library_size"]
            )

        samples["x"] = generative_dists["p_x"].sample()

        return generative_dists, samples
