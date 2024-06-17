import torch.nn as nn

from sams_vae.models.utils.predictor import PerturbationPlatedPredictor


class SAMSVAEBetaBernoulliPredictor(PerturbationPlatedPredictor):
    def __init__(self, model: nn.Module, guide: nn.Module):
        super().__init__(
            model=model,
            guide=guide,
            local_variables=["z_basal"],
            perturbation_plated_variables=["E", "mask_probs", "mask"],
        )
