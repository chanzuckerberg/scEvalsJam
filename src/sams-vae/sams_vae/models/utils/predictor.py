from typing import Dict, Iterable, Literal, Optional, Sequence, Union

import anndata
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sams_vae.analysis.average_treatment_effects import (
    estimate_model_average_treatment_effect,
)
from sams_vae.data.utils.perturbation_datamodule import PerturbationDataModule
from sams_vae.data.utils.perturbation_dataset import PerturbationDataset


class PerturbationPlatedPredictor(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        guide: nn.Module,
        local_variables: Optional[Iterable[str]] = None,
        perturbation_plated_variables: Optional[Iterable[str]] = None,
        dosage_independent_variables: Optional[Iterable[str]] = None,
    ):
        super().__init__()

        # convert variables to lists
        local_variables = list(local_variables) if local_variables is not None else []
        perturbation_plated_variables = (
            list(perturbation_plated_variables)
            if perturbation_plated_variables is not None
            else []
        )

        # check valid variable lists
        assert sorted(model.get_var_keys()) == sorted(
            guide.get_var_keys()
        ), "Mismatch in model and guide variables"

        # make sure that all variables are specified as a local variable or a
        # perturbation plated variable
        variables = local_variables + perturbation_plated_variables
        assert sorted(list(model.get_var_keys())) == sorted(
            variables
        ), "Mismatch between model variables and variables specified to loss module"

        # make sure that dosage_independent_variables are valid
        if dosage_independent_variables is not None:
            assert set(dosage_independent_variables).issubset(set(variables))

        # store passed in values
        self.model = model.eval()
        self.guide = guide.eval()
        self.local_variables = local_variables
        self.perturbation_plated_variables = perturbation_plated_variables
        self.dosage_independent_variables = dosage_independent_variables

    def _get_device(self):
        # TODO: clean up device management approach
        # assumes all parameters/buffers for model and guide are on same device
        device = next(self.model.parameters()).device
        return device

    @torch.no_grad()
    def compute_predictive_iwelbo(
        self,
        loaders: Union[DataLoader, Sequence[DataLoader]],
        n_particles: int,
    ) -> pd.DataFrame:
        """
        Compute IWELBO(X|variables, theta, phi) for trained model
        Importantly, does not score plated variables against priors

        Parameters
        ----------
        loaders: dataloaders with perturbation datasets
        n_particles: number of particles to compute predictive IWELBO

        Returns
        -------
        Dataframe with estimated predictive IWELBO for each datapoint
        in column "IWELBO", sample IDs in index

        """
        if isinstance(loaders, DataLoader):
            loaders = [loaders]

        device = self._get_device()

        # sample perturbation plated variables to share across batches
        guide_dists, guide_samples = self.guide(n_particles=n_particles)
        condition_values = {}
        for var_name in self.perturbation_plated_variables:
            condition_values[var_name] = guide_samples[var_name]

        # compute importance weighted ELBO
        id_list = []
        iwelbo_list = []
        for loader in loaders:
            idx_list_curr = []
            for batch in tqdm(loader):
                for k in batch:
                    batch[k] = batch[k].to(device)
                idx_list_curr.append(batch["idx"].detach().cpu().numpy())

                # catch adding library size if it becomes relevant
                # note: this part is not necessary for the guide
                # typically the llk is not evaluated in the guide, so we can skip this
                if self.model.likelihood_key == "library_nb":
                    condition_values["library_size"] = batch["library_size"]

                guide_dists, guide_samples = self.guide(
                    X=batch["X"],
                    D=batch["D"],
                    condition_values=condition_values,
                    n_particles=n_particles,
                )

                # catch adding library size if it becomes relevant to the likelihood
                # necessary to evaluate predictive
                # this is strictly not the elegant way to do this, that would
                # be via args/kwargs, but a quick fix
                if self.model.likelihood_key == "library_nb":
                    guide_samples["library_size"] = batch["library_size"]

                model_dists, model_samples = self.model(
                    D=batch["D"],
                    condition_values=guide_samples,
                    n_particles=n_particles,
                )

                iwelbo_terms_dict = {}
                # shape: (n_particles, n_samples)
                iwelbo_terms_dict["x"] = model_dists["p_x"].log_prob(batch["X"]).sum(-1)
                for var_name in self.local_variables:
                    p = (
                        model_dists[f"p_{var_name}"]
                        .log_prob(guide_samples[var_name])
                        .sum(-1)
                    )
                    q = (
                        guide_dists[f"q_{var_name}"]
                        .log_prob(guide_samples[var_name])
                        .sum(-1)
                    )
                    iwelbo_terms_dict[var_name] = p - q

                # shape: (n_particles, n_samples)
                iwelbo_terms = sum([v for k, v in iwelbo_terms_dict.items()])
                # compute batch IWELBO
                # shape: (n_samples,)
                batch_iwelbo = torch.logsumexp(iwelbo_terms, dim=0) - np.log(
                    n_particles
                )

                iwelbo_list.append(batch_iwelbo.detach().cpu().numpy())

            idx_curr = np.concatenate(idx_list_curr)
            dataset: PerturbationDataset = loader.dataset
            ids_curr = dataset.convert_idx_to_ids(idx_curr)
            id_list.append(ids_curr)

        iwelbo = np.concatenate(iwelbo_list)
        ids = np.concatenate(id_list)

        iwelbo_df = pd.DataFrame(
            index=ids, columns=["IWELBO"], data=iwelbo.reshape(-1, 1)
        )
        return iwelbo_df

    @torch.no_grad()
    def sample_observations(
        self,
        dosages: torch.Tensor,
        perturbation_names: Optional[Sequence[str]],
        n_particles: int = 1,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        x_var_info: Optional[pd.DataFrame] = None,
    ) -> anndata.AnnData:
        """
        Sample observations conditioned on perturbations

        Parameters
        ----------
        dosages: encoded dosages for perturbations of interest
        perturbation_names: optional string names for each row in dosages
        n_particles: number of samples to take for each dosage

        Returns
        -------
        anndata of samples dosage index, perturbation name, and particle_idx in obs,
        sampled observations in X, and x_var_info in var
        """
        device = self._get_device()
        dosages = dosages.to(device)
        # sample perturbation plated variables to share across batches
        guide_dists, guide_samples = self.guide(n_particles=n_particles)
        if condition_values is None:
            condition_values = dict()
        else:
            condition_values = {k: v.to(device) for k, v in condition_values.items()}
        for var_name in self.perturbation_plated_variables:
            condition_values[var_name] = guide_samples[var_name]

        x_samples_list = []
        for i in tqdm(range(dosages.shape[0])):
            D = dosages[i : i + 1]
            _, model_samples = self.model(
                D=D, condition_values=condition_values, n_particles=n_particles
            )
            x_samples_list.append(model_samples["x"].detach().cpu().numpy().squeeze())

        x_samples = np.concatenate(x_samples_list)
        obs = pd.DataFrame(index=np.arange(x_samples.shape[0]))
        obs["perturbation_idx"] = np.repeat(np.arange(dosages.shape[0]), n_particles)
        obs["particle_idx"] = np.tile(np.arange(dosages.shape[0]), n_particles)
        if perturbation_names is not None:
            obs["perturbation_name"] = np.array(perturbation_names)[
                obs["perturbation_idx"].to_numpy()
            ]

        adata = anndata.AnnData(obs=obs, X=x_samples)
        if x_var_info is not None:
            adata.var = x_var_info.copy()
        return adata

    def sample_observations_data_module(
        self,
        data_module: PerturbationDataModule,
        n_particles: int,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Sample observations from each unique intervention observed in a PerturbationDataModule
        TODO: come up with better naming for this method

        Parameters
        ----------
        data_module
        n_particles

        Returns
        -------
        anndata with samples from unique interventions in data module
        obs will have perturabtion name and particle idx, X will have sampled observations,
        and var dataframe will have
        """
        perturbation_names = data_module.get_unique_observed_intervention_info().index
        D = data_module.get_unique_observed_intervention_dosages(perturbation_names)
        x_var_info = data_module.get_x_var_info()

        adata = self.sample_observations(
            dosages=D,
            perturbation_names=perturbation_names,
            x_var_info=x_var_info,
            n_particles=n_particles,
            condition_values=condition_values,
        )

        return adata

    @torch.no_grad()
    def estimate_average_treatment_effects(
        self,
        dosages_alt: torch.Tensor,
        dosages_control: torch.Tensor,
        method: Literal["mean", "perturbseq"],
        n_particles: int = 1000,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        perturbation_names_alt: Optional[Sequence[str]] = None,
        perturbation_name_control: Optional[str] = None,
        x_var_info: Optional[pd.DataFrame] = None,
        batch_size: int = 500,
    ) -> anndata.AnnData:
        """
        Estimate average treatment effects of alternate dosages relative control dosage using model

        Parameters
        ----------
        dosages_alt: alternate dosages
        dosages_control: control dosage
        method: mean or perturbseq (log fold change after normalization for library size)
        n_particles: number of samples per treatment for estimate
        condition_values: any additional conditioning variables for model / guide
        perturbation_names: names for dosages, will be used as obs index if provided
        x_var_info: names of observed variables, will be included as var if provided

        Returns
        -------
        anndata with average treatment effects in X, perturbation names as obs index if provided
        (aligned to dosages_alt otherwise), and x_var_info as var if provided
        """
        device = self._get_device()
        dosages_alt = dosages_alt.to(device)
        dosages_control = dosages_control.to(device)
        if condition_values is not None:
            for k in condition_values:
                condition_values[k] = condition_values[k].to(device)

        average_treatment_effects = estimate_model_average_treatment_effect(
            model=self.model,
            guide=self.guide,
            dosages_alt=dosages_alt,
            dosages_control=dosages_control,
            n_particles=n_particles,
            method=method,
            condition_values=condition_values,
            batch_size=batch_size,
            dosage_independent_variables=self.dosage_independent_variables,
        )
        adata = anndata.AnnData(average_treatment_effects)
        if perturbation_names_alt is not None:
            adata.obs = pd.DataFrame(index=np.array(perturbation_names_alt))
        if perturbation_name_control is not None:
            adata.uns["control"] = perturbation_name_control
        if x_var_info is not None:
            adata.var = x_var_info.copy()
        return adata

    def estimate_average_effects_data_module(
        self,
        data_module: PerturbationDataModule,
        control_label: str,
        method: Literal["mean", "perturbseq"],
        n_particles: int = 1000,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: int = 500,
    ):
        perturbation_names = data_module.get_unique_observed_intervention_info().index
        perturbation_names_alt = [
            name for name in perturbation_names if name != control_label
        ]

        dosages_alt = data_module.get_unique_observed_intervention_dosages(
            perturbation_names_alt
        )
        dosages_ref = data_module.get_unique_observed_intervention_dosages(
            [control_label]
        )

        x_var_info = data_module.get_x_var_info()

        adata = self.estimate_average_treatment_effects(
            dosages_alt=dosages_alt,
            dosages_control=dosages_ref,
            method=method,
            n_particles=n_particles,
            condition_values=condition_values,
            perturbation_names_alt=perturbation_names_alt,
            perturbation_name_control=control_label,
            x_var_info=x_var_info,
            batch_size=batch_size,
        )
        return adata
