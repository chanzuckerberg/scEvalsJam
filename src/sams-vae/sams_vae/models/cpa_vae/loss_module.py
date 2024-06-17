from torch import nn

from sams_vae.models.utils.loss_modules import (
    PerturbationPlatedELBOLossModule,
    PerturbationPlatedIWELBOLossModule,
)


class CPAVAE_ELBOLossModule(PerturbationPlatedELBOLossModule):
    def __init__(self, model: nn.Module, guide: nn.Module):
        super().__init__(
            model=model,
            guide=guide,
            local_variables=["z_basal"],
            perturbation_plated_variables=["E"],
        )


class CPAVAE_IWELBOLossModule(PerturbationPlatedIWELBOLossModule):
    def __init__(self, model: nn.Module, guide: nn.Module):
        super().__init__(
            model=model,
            guide=guide,
            local_variables=["z_basal"],
            perturbation_plated_variables=["E"],
        )
