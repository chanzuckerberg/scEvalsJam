from typing import Dict, Literal, Optional, Tuple

import torch
from torch import nn

from sams_vae.data.utils.perturbation_datamodule import (
    ObservationNormalizationStatistics,
)
from sams_vae.models.utils.delta import DeltaDist
from sams_vae.models.utils.gumbel_softmax_bernoulli import (
    GumbelSoftmaxBernoulliStraightThrough,
)
from sams_vae.models.utils.mlp import get_likelihood_mlp
from sams_vae.models.utils.normalization import get_normalization_module


class SVAEPlusGuide(nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_treatments: int,
        n_phenos: int,
        mask_init_logits: float,
        encoder_n_layers: int,
        encoder_n_hidden: int,
        encoder_input_normalization: Optional[
            Literal["standardize", "log_standardize"]
        ],
        x_normalization_stats: Optional[ObservationNormalizationStatistics],
        gs_temperature: float = 1,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_treatments = n_treatments
        self.n_phenos = n_phenos

        self.encoder_input_normalization = encoder_input_normalization
        self.x_normalization_stats = x_normalization_stats

        self.param_dict = torch.nn.ParameterDict()
        self.param_dict["q_mask_logits"] = torch.nn.Parameter(
            mask_init_logits * torch.ones((n_treatments, n_latent))
        )

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
            n_input=n_phenos,
            n_output=n_latent,
            n_layers=encoder_n_layers,
            n_hidden=encoder_n_hidden,
            use_batch_norm=False,
        )

        self.register_buffer("gs_temperature", gs_temperature * torch.ones((1,)))

        self.var_eps = 1e-4

    def get_var_keys(self):
        var_keys = ["z", "mask_prob", "mask"]
        return var_keys

    def forward(
        self,
        X: Optional[torch.Tensor] = None,
        D: Optional[torch.Tensor] = None,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
    ) -> Tuple[Dict[str, torch.distributions.Distribution], Dict[str, torch.Tensor]]:
        """
        Compute q(z, mask_prob, mask | X, D) = q(mask_prob) q(mask | mask_prob) q(z | X) and sample

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

        # q(mask_prob)
        mask_probs = torch.sigmoid(self.param_dict["q_mask_logits"])
        guide_distributions["q_mask_prob"] = DeltaDist(mask_probs)

        if "mask_prob" not in condition_values:
            guide_samples["mask_prob"] = guide_distributions["q_mask_prob"].rsample(
                (n_particles,)
            )
        else:
            guide_samples["mask_prob"] = condition_values["mask_prob"]

        # q(mask | mask_prob)
        guide_distributions["q_mask"] = GumbelSoftmaxBernoulliStraightThrough(
            temperature=self.gs_temperature,
            logits=torch.logit(guide_samples["mask_prob"]),
        )

        if "mask" not in condition_values:
            # already has n_particles from mask_prob
            guide_samples["mask"] = guide_distributions["q_mask"].rsample()
        else:
            guide_samples["mask"] = condition_values["mask"]

        if X is not None and D is not None:
            # compute q(z|x)
            encoder_input = X

            # normalize input for z_basal encoder
            if self.normalization_module is not None:
                encoder_input = self.normalization_module(encoder_input)

            # expand encoder_input on dim 0 to match n_particles
            encoder_input = torch.unsqueeze(encoder_input, dim=0).expand(
                n_particles, -1, -1
            )

            guide_distributions["q_z"] = self.z_encoder(encoder_input)
            guide_samples["z"] = guide_distributions[
                "q_z"
            ].rsample()  # already n_particles

        if "z" in condition_values:
            guide_samples["z"] = condition_values["z"]

        return guide_distributions, guide_samples
