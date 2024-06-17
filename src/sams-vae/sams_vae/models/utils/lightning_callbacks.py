from typing import Any, Optional

import anndata
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from sams_vae.analysis.simulation_metrics import get_mask_stats
from sams_vae.data.utils.anndata import align_adatas
from sams_vae.models.utils.perturbation_lightning_module import (
    PerturbationLightningModule,
)


# Callbacks for training
class GradientNormTracker(pl.Callback):
    def __init__(self, norm_type: float = 2, every_n_steps: int = 100):
        self.norm_type = norm_type
        self.every_n_steps = every_n_steps

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.every_n_steps != 0:
            return

        grads = [p.grad for p in pl_module.parameters() if p.grad is not None]
        device = grads[0].device
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(g.detach(), self.norm_type).to(device) for g in grads]
            ),
            self.norm_type,
        )
        pl_module.log("train/gradient_norm", total_norm)


class TreatmentMaskStatsTracker(pl.Callback):
    def __init__(
        self,
        mask_key: str = "mask",
        true_latent_effects: Optional[anndata.AnnData] = None,
        d_var: Optional[pd.DataFrame] = None,
        n_particles: int = 100,
    ):
        self.mask_key = mask_key
        self.true_latent_effects = true_latent_effects
        self.d_var = d_var
        self.n_particles = n_particles

    def on_validation_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: PerturbationLightningModule,
    ) -> None:
        if pl_module.predictor is None:
            return

        guide_dists, guide_samples = pl_module.loss_module.guide(
            n_particles=self.n_particles
        )

        if self.mask_key not in guide_samples:
            return

        mask_dist = guide_dists[f"q_{self.mask_key}"]
        mask_samples = guide_samples[self.mask_key]
        if hasattr(mask_dist, "probs"):
            # directly access estimated probabilities
            mask_freq = mask_dist.probs.detach()
            if len(mask_freq.shape) == 3:
                # take average over n_particles
                mask_freq = mask_freq.mean(0)
        else:
            # use observed frequency from samples
            mask_freq = mask_samples.mean(0)

        pl_module.log(
            "val/frac_mask_freq_geq_0.5",
            (mask_freq >= 0.5).type(torch.FloatTensor).mean(),
        )
        pl_module.log(
            "val/frac_mask_freq_geq_0.8",
            (mask_freq >= 0.8).type(torch.FloatTensor).mean(),
        )
        pl_module.log(
            "val/frac_mask_freq_leq_0.2",
            (mask_freq <= 0.2).type(torch.FloatTensor).mean(),
        )

        if self.true_latent_effects is not None and self.d_var is not None:
            inferred_mask = (mask_freq.detach().cpu().numpy() > 0.5).astype(np.float32)
            inferred_adata = anndata.AnnData(obs=self.d_var.copy(), X=inferred_mask)

            true_adata, inferred_adata = align_adatas(
                self.true_latent_effects, inferred_adata
            )

            true_mask = true_adata.X != 0
            inferred_mask = inferred_adata.X
            stats, idx = get_mask_stats(true_mask, inferred_mask)
            for k, v in stats.items():
                pl_module.log(f"val/mask_{k}", v.item())
