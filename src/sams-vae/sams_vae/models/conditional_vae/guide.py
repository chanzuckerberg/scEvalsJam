from typing import Dict, Literal, Optional, Tuple

import torch
from torch import nn

from sams_vae.data.utils.perturbation_datamodule import (
    ObservationNormalizationStatistics,
)
from sams_vae.models.utils.mlp import get_likelihood_mlp
from sams_vae.models.utils.normalization import get_normalization_module


class ConditionalVAEGuide(nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_treatments: int,
        n_phenos: int,
        encoder_n_layers: int,
        encoder_n_hidden: int,
        encoder_input_normalization: Optional[
            Literal["standardize", "log_standardize"]
        ],
        x_normalization_stats: Optional[ObservationNormalizationStatistics],
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_treatments = n_treatments
        self.n_phenos = n_phenos

        self.encoder_input_normalization = encoder_input_normalization
        self.x_normalization_stats = x_normalization_stats

        if self.encoder_input_normalization is None:
            self.normalization_module = None
        else:
            assert x_normalization_stats is not None, "Missing x_normalization_stats"
            self.normalization_module = get_normalization_module(
                key=self.encoder_input_normalization,
                normalization_stats=x_normalization_stats,
            )

        self.z_encoder = get_likelihood_mlp(
            likelihood_key="normal",
            n_input=n_phenos + n_treatments,
            n_output=n_latent,
            n_layers=encoder_n_layers,
            n_hidden=encoder_n_hidden,
            use_batch_norm=False,
        )

    def get_var_keys(self):
        var_keys = ["z"]
        return var_keys

    def forward(
        self,
        X: Optional[torch.Tensor] = None,
        D: Optional[torch.Tensor] = None,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
    ) -> Tuple[Dict[str, torch.distributions.Distribution], Dict[str, torch.Tensor]]:
        """
        Compute q(z | X, D) and sample

        Parameters
        ----------
        X: observations
        D: perturbation dosages
        condition_values: values for random variables to condition on
        n_particles: number of samples to take from q

        Returns
        -------
        Tuple of guide distribution dict and guide samples dict
        Each has string keys for variable names
        """
        if condition_values is None:
            condition_values = dict()

        guide_distributions: Dict[str, torch.distributions.Distribution] = {}
        guide_samples: Dict[str, torch.Tensor] = {}

        if X is not None and D is not None:
            encoder_input = X

            # normalize input for z_basal encoder
            if self.normalization_module is not None:
                encoder_input = self.normalization_module(encoder_input)

            # concatenate dosages ( n x (n_phenos + n_latent) )
            encoder_input = torch.cat([encoder_input, D], dim=-1)

            guide_distributions["q_z"] = self.z_encoder(encoder_input)
            guide_samples["z"] = guide_distributions["q_z"].rsample((n_particles,))

        if "z" in condition_values:
            guide_samples["z"] = condition_values["z"]

        return guide_distributions, guide_samples
