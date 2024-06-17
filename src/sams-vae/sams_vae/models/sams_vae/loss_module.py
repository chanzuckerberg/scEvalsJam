from typing import Dict, Iterable, Optional

from torch import nn

from sams_vae.models.utils.loss_modules import (
    PerturbationPlatedELBOCustomReweightedLossModule,
    PerturbationPlatedELBOLossModule,
    PerturbationPlatedIWELBOLossModule,
)


class SAMSVAE_ELBOLossModule(PerturbationPlatedELBOLossModule):
    def __init__(self, model: nn.Module, guide: nn.Module):
        super().__init__(
            model=model,
            guide=guide,
            local_variables=["z_basal"],
            perturbation_plated_variables=["E", "mask"],
        )


class SAMSVAE_CustomReweightedELBOLossModule(
    PerturbationPlatedELBOCustomReweightedLossModule
):
    def __init__(
        self,
        model: nn.Module,
        guide: nn.Module,
        custom_prior_weights: Optional[Dict[str, float]] = None,
        custom_plated_prior_additional_weight_proportional_n: Optional[
            Dict[str, float]
        ] = None,
        custom_loss_term_weights: Optional[Iterable[str]] = None,
        custom_plated_loss_term_additional_weight_proportional_n: Optional[
            Dict[str, float]
        ] = None,
    ):
        super().__init__(
            model=model,
            guide=guide,
            local_variables=["z_basal"],
            perturbation_plated_variables=["E", "mask"],
            custom_prior_weights=custom_prior_weights,
            custom_plated_prior_additional_weight_proportional_n=custom_plated_prior_additional_weight_proportional_n,  # noqa: E501
            custom_loss_term_weights=custom_loss_term_weights,
            custom_plated_loss_term_additional_weight_proportional_n=custom_plated_loss_term_additional_weight_proportional_n,  # noqa: E501
        )


class SAMSVAE_IWELBOLossModule(PerturbationPlatedIWELBOLossModule):
    def __init__(self, model: nn.Module, guide: nn.Module):
        super().__init__(
            model=model,
            guide=guide,
            local_variables=["z_basal"],
            perturbation_plated_variables=["E", "mask"],
        )
