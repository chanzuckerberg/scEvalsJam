from typing import Dict, Literal, Optional, Tuple

import torch
from torch import nn

from sams_vae.data.utils.perturbation_datamodule import (
    ObservationNormalizationStatistics,
)
from sams_vae.models.utils.gumbel_softmax_bernoulli import (
    GumbelSoftmaxBernoulliStraightThrough,
)
from sams_vae.models.utils.mlp import get_likelihood_mlp
from sams_vae.models.utils.normalization import get_normalization_module


class SAMSVAECorrelatedNormalGuide(nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_treatments: int,
        n_phenos: int,
        basal_encoder_n_layers: int,
        basal_encoder_n_hidden: int,
        basal_encoder_input_normalization: Optional[
            Literal["standardize", "log_standardize"]
        ],
        embedding_encoder_n_layers: int,
        embedding_encoder_n_hidden: int,
        x_normalization_stats: Optional[ObservationNormalizationStatistics],
        mask_init_logits: float = 0,
        gs_temperature: float = 1,
        mean_field_encoder: bool = False,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_treatments = n_treatments
        self.n_phenos = n_phenos

        self.basal_encoder_input_normalization = basal_encoder_input_normalization
        self.x_normalization_stats = x_normalization_stats

        self.mean_field_encoder = mean_field_encoder

        self.param_dict = torch.nn.ParameterDict()
        # q(M) parameters
        self.param_dict["q_mask_logits"] = torch.nn.Parameter(
            mask_init_logits * torch.ones((n_treatments, n_latent))
        )

        # q(E|M) parameters
        self.embedding_encoder = get_likelihood_mlp(
            likelihood_key="normal",
            n_input=n_latent + n_treatments,
            n_output=n_latent,
            n_layers=embedding_encoder_n_layers,
            n_hidden=embedding_encoder_n_hidden,
            use_batch_norm=False,
        )

        self.register_buffer("treatment_one_hot", torch.eye(n_treatments))

        if self.basal_encoder_input_normalization is None:
            self.normalization_module = None
        else:
            assert x_normalization_stats is not None, "Missing x_normalization_stats"
            self.normalization_module = get_normalization_module(
                key=self.basal_encoder_input_normalization,
                normalization_stats=x_normalization_stats,
            )

        # q(z^b | X, D, E, M) parameters
        self.z_basal_encoder = get_likelihood_mlp(
            likelihood_key="normal",
            n_input=n_phenos if mean_field_encoder else n_phenos + n_latent,
            n_output=n_latent,
            n_layers=basal_encoder_n_layers,
            n_hidden=basal_encoder_n_hidden,
            use_batch_norm=False,
        )

        self.register_buffer("gs_temperature", gs_temperature * torch.ones((1,)))

        self.var_eps = 1e-4

    def get_var_keys(self):
        var_keys = ["z_basal", "E", "mask"]
        return var_keys

    def forward(
        self,
        X: Optional[torch.Tensor] = None,
        D: Optional[torch.Tensor] = None,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
    ) -> Tuple[Dict[str, torch.distributions.Distribution], Dict[str, torch.Tensor]]:
        """
        Compute q(z_basal, M, E | X, D) = q(M) q(E|M) q(z_basal | M, E, X, D) and sample

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

        # q(M)
        guide_distributions["q_mask"] = GumbelSoftmaxBernoulliStraightThrough(
            temperature=self.gs_temperature,
            logits=self.param_dict["q_mask_logits"],
        )

        if "mask" not in condition_values:
            guide_samples["mask"] = guide_distributions["q_mask"].rsample(
                (n_particles,)
            )
        else:
            guide_samples["mask"] = condition_values["mask"]

        # q(E|M)
        # provide mask and 1-hot encoding of perturbation index to embedding encoder
        treatment_one_hot = self.treatment_one_hot.unsqueeze(0).expand(
            n_particles, -1, -1
        )
        embedding_encoder_input = torch.cat(
            [guide_samples["mask"], treatment_one_hot], dim=-1
        )
        guide_distributions["q_E"] = self.embedding_encoder(embedding_encoder_input)

        if "E" not in condition_values:
            # q(E|M) already has n_particles from mask
            guide_samples["E"] = guide_distributions["q_E"].rsample()
        else:
            guide_samples["E"] = condition_values["E"]

        if X is not None and D is not None:
            # compute q(z_basal|x) if mean field encoder, q(z_basal | x, M, E) if not
            encoder_input = X

            # normalize input for z_basal encoder
            if self.normalization_module is not None:
                encoder_input = self.normalization_module(encoder_input)

            # expand encoder_input on dim 0 to match n_particles
            encoder_input = torch.unsqueeze(encoder_input, dim=0).expand(
                n_particles, -1, -1
            )

            if not self.mean_field_encoder:
                # q(z_basal|x, M, E) by concatenating estimated latent offsets to x
                latent_offset = torch.matmul(
                    D, guide_samples["mask"] * guide_samples["E"]
                )
                encoder_input = torch.cat([encoder_input, latent_offset], dim=-1)

            guide_distributions["q_z_basal"] = self.z_basal_encoder(encoder_input)
            guide_samples["z_basal"] = guide_distributions[
                "q_z_basal"
            ].rsample()  # already n_particles

        if "z_basal" in condition_values:
            guide_samples["z_basal"] = condition_values["z_basal"]

        return guide_distributions, guide_samples
