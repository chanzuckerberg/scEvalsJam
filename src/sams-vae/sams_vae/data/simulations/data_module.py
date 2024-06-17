from typing import Literal, Optional, Sequence

import anndata
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sams_vae.analysis.average_treatment_effects import (
    estimate_data_average_treatment_effects,
)
from sams_vae.data.simulations.sams_vae_simulation import simulate_data_sams_vae
from sams_vae.data.simulations.svae_plus_paper_simulation import (
    simulate_svae_plus_paper_dataset,
)
from sams_vae.data.utils.perturbation_datamodule import (
    ObservationNormalizationStatistics,
    PerturbationDataModule,
)
from sams_vae.data.utils.perturbation_dataset import TensorPerturbationDataset


class SAMSVAESimulationDataModule(PerturbationDataModule):
    def __init__(
        self,
        n_latent: int = 15,
        n_treatments: int = 100,
        n_phenos: int = 100,
        n_hidden_layers_decoder: int = 4,
        n_hidden_dim_decoder: int = 50,
        decoder_exp_output: bool = False,
        decoder_layer_sparsity: Optional[float] = None,
        decoder_init_strategy: str = "orthogonal",
        mask_prior_prob: float = 2 / 15,
        latent_effect_magnitude_mean: float = 5,
        latent_effect_magnitude_std: float = 0.5,
        n_samples_per_treatment_train: int = 500,
        n_samples_per_treatment_val: int = 100,
        n_samples_per_treatment_test: int = 100,
        frac_var_pheno_noise: float = 0.2,
        seed: int = 0,
        batch_size: int = 128,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.adata, self.decoder = simulate_data_sams_vae(
            n_latent=n_latent,
            n_treatments=n_treatments,
            n_phenos=n_phenos,
            n_hidden_layers_decoder=n_hidden_layers_decoder,
            n_hidden_dim_decoder=n_hidden_dim_decoder,
            decoder_exp_output=decoder_exp_output,
            decoder_init_strategy=decoder_init_strategy,
            decoder_layer_sparsity=decoder_layer_sparsity,
            mask_prior_prob=mask_prior_prob,
            latent_effect_magnitude_mean=latent_effect_magnitude_mean,
            latent_effect_magnitude_std=latent_effect_magnitude_std,
            n_samples_per_treatment_train=n_samples_per_treatment_train,
            n_samples_per_treatment_val=n_samples_per_treatment_val,
            n_samples_per_treatment_test=n_samples_per_treatment_test,
            frac_var_pheno_noise=frac_var_pheno_noise,
            seed=seed,
            return_decoder=True,
        )

        self.d_var_info = self.adata.uns["D_var_info"]
        D = torch.from_numpy(self.adata.obsm["D"].astype(np.float32))
        X = torch.from_numpy(self.adata.X.copy())

        ids_tr = self.adata.obs[self.adata.obs["split"] == "train"].index
        X_tr = X[(self.adata.obs["split"] == "train").to_numpy()]
        D_tr = D[(self.adata.obs["split"] == "train").to_numpy()]

        ids_val = self.adata.obs[self.adata.obs["split"] == "val"].index
        X_val = X[(self.adata.obs["split"] == "val").to_numpy()]
        D_val = D[(self.adata.obs["split"] == "val").to_numpy()]

        ids_test = self.adata.obs[self.adata.obs["split"] == "test"].index
        X_test = X[(self.adata.obs["split"] == "test").to_numpy()]
        D_test = D[(self.adata.obs["split"] == "test").to_numpy()]

        self.train_dataset = TensorPerturbationDataset(X=X_tr, D=D_tr, ids=ids_tr)
        self.val_dataset = TensorPerturbationDataset(X=X_val, D=D_val, ids=ids_val)
        self.test_dataset = TensorPerturbationDataset(X=X_test, D=D_test, ids=ids_test)

        x_tr_mean = X_tr.mean(0)
        x_tr_std = X_tr.std(0)
        log_x_tr = torch.log(X_tr + 1)
        log_x_tr_mean = log_x_tr.mean(0)
        log_x_tr_std = log_x_tr.std(0)

        self.x_train_statistics = ObservationNormalizationStatistics(
            x_mean=x_tr_mean,
            x_std=x_tr_std,
            log_x_mean=log_x_tr_mean,
            log_x_std=log_x_tr_std,
        )

        # because there are no perturbation combinations in this simulation,
        # unique_perturbations are the same as the observed perturbations
        self.d_var_info["train"] = self.get_train_perturbation_obs_counts().numpy() > 0
        self.d_var_info["val"] = self.get_val_perturbation_obs_counts().numpy() > 0
        self.d_var_info["test"] = self.get_test_perturbation_obs_counts().numpy() > 0

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_train_perturbation_obs_counts(self) -> torch.Tensor:
        return self.train_dataset.get_dosage_obs_per_dim()

    def get_val_perturbation_obs_counts(self) -> torch.Tensor:
        return self.val_dataset.get_dosage_obs_per_dim()

    def get_test_perturbation_obs_counts(self) -> torch.Tensor:
        return self.test_dataset.get_dosage_obs_per_dim()

    def get_x_var_info(self) -> pd.DataFrame:
        return self.adata.var.copy()

    def get_d_var_info(self) -> pd.DataFrame:
        return self.d_var_info.copy()

    def get_obs_info(self) -> pd.DataFrame:
        return self.adata.obs.copy()

    def get_x_train_statistics(self) -> ObservationNormalizationStatistics:
        return self.x_train_statistics

    def get_unique_observed_intervention_info(self) -> pd.DataFrame:
        return self.d_var_info.copy()

    def get_unique_observed_intervention_dosages(
        self, pert_names: Sequence
    ) -> torch.Tensor:
        # TODO: clean up to use D_unique in adata
        D = torch.zeros((len(pert_names), self.d_var_info.shape[0]))
        pert_idx_map = dict(
            zip(self.d_var_info.index, list(range(self.d_var_info.shape[0])))
        )
        for i, pert_name in enumerate(pert_names):
            D[i, pert_idx_map[pert_name]] = 1
        return D

    def get_estimated_average_treatment_effects(
        self,
        method: Literal["mean", "perturbseq"],
        split: Optional[str] = None,
    ) -> Optional[anndata.AnnData]:
        # TODO: anndata does implicit conversion to string which
        #  causes issue if index is not string
        adata = self.adata
        if split is not None:
            adata = adata[adata.obs["split"] == split]
        return estimate_data_average_treatment_effects(
            adata, label_col="treatment", control_label="0", method=method
        )

    def get_simulated_latent_effects(self) -> Optional[anndata.AnnData]:
        return anndata.AnnData(
            obs=self.d_var_info.copy(),
            X=self.adata.uns["latent_effects"],
        )


class SVAEPaperSimulationDataModule(PerturbationDataModule):
    def __init__(
        self,
        n_perturbations: int = 100,
        n_latent: int = 15,
        n_phenos: int = 100,
        latent_shift_magnitude_mean: float = 5.0,
        bernoulli_mask_prob: Optional[float] = None,
        n_samples_per_perturbation_train: int = 500,
        n_samples_per_perturbation_val: int = 100,
        n_samples_per_perturbation_test: int = 100,
        batch_size: int = 128,
    ):
        super().__init__()
        self.batch_size = batch_size

        # load simulation data
        self.adata = simulate_svae_plus_paper_dataset(
            n_cells_per_chem_tr=n_samples_per_perturbation_train,
            n_cells_per_chem_val=n_samples_per_perturbation_val,
            n_cells_per_chem_test=n_samples_per_perturbation_test,
            n_chem=n_perturbations,
            n_latent=n_latent,
            n_genes=n_phenos,
            mean_shift_size=latent_shift_magnitude_mean,
            bernoulli_mask_prob=bernoulli_mask_prob,
        )

        # encode dosages
        dosage_df = pd.get_dummies(self.adata.obs["T"])
        self.d_var_info = dosage_df.T[[]]
        D = torch.from_numpy(dosage_df.to_numpy().astype(np.float32))

        X = torch.from_numpy(self.adata.X.copy())

        ids_tr = self.adata.obs[self.adata.obs["split"] == "train"].index
        X_tr = X[(self.adata.obs["split"] == "train").to_numpy()]
        D_tr = D[(self.adata.obs["split"] == "train").to_numpy()]

        ids_val = self.adata.obs[self.adata.obs["split"] == "val"].index
        X_val = X[(self.adata.obs["split"] == "val").to_numpy()]
        D_val = D[(self.adata.obs["split"] == "val").to_numpy()]

        ids_test = self.adata.obs[self.adata.obs["split"] == "test"].index
        X_test = X[(self.adata.obs["split"] == "test").to_numpy()]
        D_test = D[(self.adata.obs["split"] == "test").to_numpy()]

        self.train_dataset = TensorPerturbationDataset(X=X_tr, D=D_tr, ids=ids_tr)
        self.val_dataset = TensorPerturbationDataset(X=X_val, D=D_val, ids=ids_val)
        self.test_dataset = TensorPerturbationDataset(X=X_test, D=D_test, ids=ids_test)

        x_tr_mean = X_tr.mean(0)
        x_tr_std = X_tr.std(0)
        log_x_tr = torch.log(X_tr + 1)
        log_x_tr_mean = log_x_tr.mean(0)
        log_x_tr_std = log_x_tr.std(0)

        self.x_train_statistics = ObservationNormalizationStatistics(
            x_mean=x_tr_mean,
            x_std=x_tr_std,
            log_x_mean=log_x_tr_mean,
            log_x_std=log_x_tr_std,
        )

        # because there are no perturbation combinations in this simulation,
        # unique_perturbations are the same as the observed perturbations
        self.d_var_info["train"] = self.get_train_perturbation_obs_counts().numpy() > 0
        self.d_var_info["val"] = self.get_val_perturbation_obs_counts().numpy() > 0
        self.d_var_info["test"] = self.get_test_perturbation_obs_counts().numpy() > 0

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_train_perturbation_obs_counts(self) -> torch.Tensor:
        return self.train_dataset.get_dosage_obs_per_dim()

    def get_val_perturbation_obs_counts(self) -> torch.Tensor:
        return self.val_dataset.get_dosage_obs_per_dim()

    def get_test_perturbation_obs_counts(self) -> torch.Tensor:
        return self.test_dataset.get_dosage_obs_per_dim()

    def get_x_var_info(self) -> pd.DataFrame:
        return self.adata.var.copy()

    def get_d_var_info(self) -> pd.DataFrame:
        return self.d_var_info.copy()

    def get_obs_info(self) -> pd.DataFrame:
        return self.adata.obs.copy()

    def get_x_train_statistics(self) -> ObservationNormalizationStatistics:
        return self.x_train_statistics

    def get_unique_observed_intervention_info(self) -> pd.DataFrame:
        return self.d_var_info.copy()

    def get_unique_observed_intervention_dosages(
        self, pert_names: Sequence
    ) -> torch.Tensor:
        D = torch.zeros((len(pert_names), self.d_var_info.shape[0]))
        pert_idx_map = dict(
            zip(self.d_var_info.index, list(range(self.d_var_info.shape[0])))
        )
        for i, pert_name in enumerate(pert_names):
            D[i, pert_idx_map[pert_name]] = 1
        return D

    def get_estimated_average_treatment_effects(
        self,
        method: Literal["mean", "perturbseq"],
        split: Optional[str] = None,
    ) -> Optional[anndata.AnnData]:
        # TODO: anndata does implicit conversion to string which
        #  causes issue if index is not string
        adata = self.adata
        if split is not None:
            adata = adata[adata.obs["split"] == split]
        return estimate_data_average_treatment_effects(
            adata, label_col="T", control_label="0", method=method
        )

    def get_simulated_latent_effects(self) -> Optional[anndata.AnnData]:
        return anndata.AnnData(
            obs=pd.DataFrame(index=self.adata.uns["prior_mean_idx"]),
            X=self.adata.uns["prior_mean"],
        )
