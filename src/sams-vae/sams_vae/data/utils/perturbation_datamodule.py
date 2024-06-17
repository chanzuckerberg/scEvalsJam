from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import anndata
import pandas as pd
import pytorch_lightning as pl
import torch


@dataclass
class ObservationNormalizationStatistics:
    x_mean: Optional[torch.Tensor] = None
    x_std: Optional[torch.Tensor] = None
    log_x_mean: Optional[torch.Tensor] = None
    log_x_std: Optional[torch.Tensor] = None


class PerturbationDataModule(pl.LightningDataModule):
    def get_train_perturbation_obs_counts(self) -> torch.Tensor:
        """
        Returns tensor with shape (n_perturbation,) that contains the
        number of samples in the training set with non-zero dosages
        for each perturbation

        Used for reweighting plated variables in ELBO
        """
        raise NotImplementedError

    def get_val_perturbation_obs_counts(self) -> torch.Tensor:
        """
        Returns tensor with shape (n_perturbation,) that contains the
        number of samples in the validation set with non-zero dosages
        for each perturbation

        Used for reweighting plated variables in ELBO
        """
        raise NotImplementedError

    def get_test_perturbation_obs_counts(self) -> torch.Tensor:
        """
        Returns tensor with shape (n_perturbation,) that contains the
        number of samples in the test set with non-zero dosages
        for each perturbation

        Used for reweighting plated variables in ELBO
        """
        raise NotImplementedError

    def get_x_var_info(self) -> pd.DataFrame:
        """
        Returns dataframe describing observation features (each row describes one column of X)
        index should be feature names
        """
        raise NotImplementedError

    def get_d_var_info(self) -> pd.DataFrame:
        """
        Return dataframe describing dosage features (each row describes one column of D)
        index should be perturbation names (strings)
        """
        raise NotImplementedError

    def get_obs_info(self) -> pd.DataFrame:
        """
        Returns dataframe describing samples (each row describes one row of X and D)
        index should be sample IDs (strings)
        """
        raise NotImplementedError

    def get_x_train_statistics(self) -> ObservationNormalizationStatistics:
        """
        Returns statistics of training data, which may be used for normalization during training
        """
        raise NotImplementedError

    def get_unique_observed_intervention_info(self) -> pd.DataFrame:
        """
        Returns dataframe with info about observed unique interventions (combinations of
        perturbations should be treated as distinct observed interventions)
        Index should have names, and should have columns for train / val / test with
        boolean if perturbation is observed in each split
        """
        raise NotImplementedError

    def get_unique_observed_intervention_dosages(
        self, pert_names: Sequence
    ) -> torch.Tensor:
        """
        Encodes unique intervention names (as returned by get_unique_observed_intervention_info)
        to perturbation dosages
        """
        raise NotImplementedError

    def get_estimated_average_treatment_effects(
        self,
        method: Literal["mean", "perturbseq"],
        split: Optional[str] = None,
    ) -> Optional[anndata.AnnData]:
        """
        Returns anndata with estimated average treatment effects from data.
        obs index contains unique observed interventions (as used in
         get_unique_observed_intervention_info and unique_observed_intervention_to_dosages)
        uns["control"] contains control intervention name
        X contains estimated average treatment effects
        var index contains observation feature names (should match index from get_var_info)

        Method specifies method to compute average treatment effect
        Mean assess average change in mean, perturbseq assess average change in mean
        after normalizing for library size and applying log transform
        """
        return None

    def get_simulated_latent_effects(self) -> Optional[anndata.AnnData]:
        """
        Returns (n_perturbations, n_latent) true latent effects for simulated data
        obs index contains contains perturbations
        """
        return None
