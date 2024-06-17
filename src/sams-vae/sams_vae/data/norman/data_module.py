from typing import Dict, Literal, Optional, Sequence

import anndata
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from sams_vae.analysis.average_treatment_effects import (
    estimate_data_average_treatment_effects,
)
from sams_vae.data.norman.download import download_norman_dataset
from sams_vae.data.utils.batch_statistics import batch_log_mean, batch_log_std
from sams_vae.data.utils.perturbation_datamodule import (
    ObservationNormalizationStatistics,
    PerturbationDataModule,
)
from sams_vae.data.utils.perturbation_dataset import SCRNASeqTensorPerturbationDataset


class BaseNormanDataModule(PerturbationDataModule):
    def __init__(
        self,
        split_kwargs: Dict,
        encode_combos_as_unique: bool = False,
        batch_size: int = 128,
        highly_variable_genes_only: bool = False,
        data_path: Optional[str] = None,
    ):
        """
        Base data module for Norman dataset tasks (OOD combos and data efficiency)
        Implements all functionality except data splitting, which is implemented
        by each subclass through `get_split_labels`


        Parameters
        ----------
        split_kwargs: dictionary of kwargs passed to `get_split_labels`
        encode_combos_as_unique: if True, encodes combinations as new perturbations
        rather than sum of individual dosages in the combo
        batch_size: batch size for data loaders
        highly_variable_genes_only: filter dataset to highly variable genes
        data_path: path of Norman anndata. If not provided, downloads to current
        directory
        """
        super().__init__()

        self.encode_combos_as_unique = encode_combos_as_unique
        self.batch_size = batch_size

#        if data_path is None:
 #           data_path = download_norman_dataset()

        self.adata = anndata.read_h5ad("/lustre/scratch126/cellgen/team298/vm11/LEARN/scEvalsJam/sams-vae/1gene-norman-split_integers.h5ad")
        if highly_variable_genes_only:
            self.adata = self.adata[:, self.adata.var["highly_variable"].to_numpy()]
        self.adata.X = self.adata.layers["counts"]
        del self.adata.layers

        guide_one_hot_cols = get_guide_one_hot_cols(self.adata.obs)
        self.adata.obs["num_guides"] = self.adata.obs[guide_one_hot_cols].sum(1)

        # generate splits (implemented in subclass)
#        split_labels = self._get_split_labels(self.adata.obs, **split_kwargs)
 #       self.adata.obs["split"] = split_labels

        # get scRNA-seq observation matrix
        X = torch.from_numpy(self.adata.X.toarray())

        # encode perturbation dosages
        treatment_labels = self.adata.obs["guide_identity"].astype(str)
        treatment_labels[
            self.adata.obs[guide_one_hot_cols].sum(1) == 0
        ] = "non-targeting"
        self.adata.obs["treatment"] = treatment_labels
        if not self.encode_combos_as_unique:
            # combinations encoded as application of two individual guides
            D = self.adata.obs[guide_one_hot_cols].to_numpy().astype(np.float32)
            self.d_var_info = pd.DataFrame(index=guide_one_hot_cols)
            D = torch.from_numpy(D)
        else:
            # combinations encoded as new treatments
            D_df = pd.get_dummies(self.adata.obs["treatment"])
            # encode non-targeting as no perturbation for consistency with other encoding
            D_df = D_df.drop(columns=["non-targeting"])
            self.d_var_info = D_df.T[[]]
            D = torch.from_numpy(D_df.to_numpy().astype(np.float32))

        # generate datasets
        ids_tr = self.adata.obs[self.adata.obs["split"] == "train"].index
        X_tr = X[(self.adata.obs["split"] == "train").to_numpy()]
        D_tr = D[(self.adata.obs["split"] == "train").to_numpy()]

        ids_val = self.adata.obs[self.adata.obs["split"] == "val"].index
        X_val = X[(self.adata.obs["split"] == "val").to_numpy()]
        D_val = D[(self.adata.obs["split"] == "val").to_numpy()]

        ids_test = self.adata.obs[self.adata.obs["split"] == "test"].index
        X_test = X[(self.adata.obs["split"] == "test").to_numpy()]
        D_test = D[(self.adata.obs["split"] == "test").to_numpy()]

        self.train_dataset = SCRNASeqTensorPerturbationDataset(
            X=X_tr, D=D_tr, ids=ids_tr
        )
        self.val_dataset = SCRNASeqTensorPerturbationDataset(
            X=X_val, D=D_val, ids=ids_val
        )
        self.test_dataset = SCRNASeqTensorPerturbationDataset(
            X=X_test, D=D_test, ids=ids_test
        )

        # compute normalization statistics
        x_tr_mean = X_tr.mean(0)
        x_tr_std = X_tr.std(0)
        log_x_tr_mean = batch_log_mean(X_tr)
        log_x_tr_std = batch_log_std(X_tr, log_x_tr_mean)

        self.x_train_statistics = ObservationNormalizationStatistics(
            x_mean=x_tr_mean,
            x_std=x_tr_std,
            log_x_mean=log_x_tr_mean,
            log_x_std=log_x_tr_std,
        )

        # generate unique intervention info dataframe
        df = self.adata.obs.groupby("treatment")["split"].agg(set).reset_index()
        for split in ["train", "val", "test"]:
            df[split] = df["split"].apply(lambda x: split in x)
        df = df.set_index("treatment").drop(columns=["split"])
        self.unique_observed_intervention_df = df

        # generate mapping from intervention names to dosages
        self.adata.obs["i"] = np.arange(self.adata.shape[0])
        idx_map = (
            self.adata.obs.drop_duplicates("treatment")
            .set_index("treatment")["i"]
            .to_dict()
        )
        self.unique_intervention_dosage_map = {k: D[v] for k, v in idx_map.items()}

    @staticmethod
    def _get_split_labels(
        obs: pd.DataFrame,
        split_seed: int,
        # used only by OOD data module
        frac_combinations_train: float = 0,
        frac_combinations_test: float = 0.2,
        # used only by data efficiency data module
        frac_combination_cells_train: float = 0,
    ) -> pd.Series:
        """
        Returns split labels for each cell (series with ID as index, aligned with obs)
        Implemented by subclasses
        """
        raise NotImplementedError

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
        return self.unique_observed_intervention_df.copy()

    def get_unique_observed_intervention_dosages(
        self, pert_names: Sequence
    ) -> torch.Tensor:
        D = torch.zeros((len(pert_names), self.d_var_info.shape[0]))
        for i, pert_name in enumerate(pert_names):
            D[i] = self.unique_intervention_dosage_map[pert_name]
        return D

    def get_estimated_average_treatment_effects(
        self, method: Literal["mean", "perturbseq"], split: Optional[str] = None
    ) -> Optional[anndata.AnnData]:
        adata = self.adata
        if split is not None:
            adata = adata[adata.obs["split"] == split]
        return estimate_data_average_treatment_effects(
            adata,
            label_col="treatment",
            control_label="non-targeting",
            method=method,
        )

    def get_simulated_latent_effects(self) -> Optional[anndata.AnnData]:
        return None


class NormanOODCombinationDataModule(BaseNormanDataModule):
    def __init__(
        self,
        frac_combinations_train: float,
        frac_combinations_test: float = 0.2,
        split_seed: int = 0,
        encode_combos_as_unique: bool = False,
        batch_size: int = 128,
        highly_variable_genes_only: bool = False,
        data_path: Optional[str] = None,
    ):
        """
        Data module to assess out of distribution generalization to new combinations
        in the Norman dataset

        The 132 unique combinations (2 targeting CRISPR guides) are randomly shuffled
        using `split_seed`. A subset of cells with  `frac_combinations_train` of the combinations
        are included in the train / val splits. A subset of cells with the last
        `frac_combinations_test` of the combinations are included in the test set

        The train / val splits additionally include cells that received 0 or 1
        targeting CRISPR guides

        Therefore, if `frac_combinations_train` < 1 - `frac_combinations_test`, the test set will
        consist of cells that received held out combinations. Additionally, as
        `frac_combinations_train` is increased, additional combinations are added to the
        train / val splits (superset of splits with smaller values).


        Parameters
        ----------
        frac_combinations_train: fraction of combinations to include in train / val set
        frac_combinations_test: fraction of combinations to include in test set
        split_seed: seed used to shuffle / split combinations
        encode_combos_as_unique: if True, represents combinations as new 1-hot perturbation
        batch_size: batch size for data loader
        highly_variable_genes_only: filter dataset to highly variable genes
        data_path: path to Norman anndata
        """
        split_kwargs = dict(
            frac_combinations_train=frac_combinations_train,
            frac_combinations_test=frac_combinations_test,
            split_seed=split_seed,
        )
        super().__init__(
            split_kwargs=split_kwargs,
            encode_combos_as_unique=encode_combos_as_unique,
            batch_size=batch_size,
            highly_variable_genes_only=highly_variable_genes_only,
            data_path=data_path,
        )

    @staticmethod
    def _get_split_labels(
        obs: pd.DataFrame,
        split_seed: int,
        # used only by OOD data module
        frac_combinations_train: float = 0,
        frac_combinations_test: float = 0.2,
        # used only by data efficiency data module
        frac_combination_cells_train: float = 0,
    ):
        # TODO: how to implement cleanly? Mypy error for changing signature
        # from superclass
        combo_guide_identities = np.sort(
            obs[obs["num_guides"] == 2]["guide_identity"].astype(str).unique()
        )

        # randomly shuffle combo guide identities using split seed
        rng = np.random.default_rng(split_seed)
        rng.shuffle(combo_guide_identities)

        # select first frac_combinations_train combos for train/val sets
        num_train_combos = int(frac_combinations_train * len(combo_guide_identities))
        train_combos = combo_guide_identities[:num_train_combos]

        # select last frac_combinations_test combos for test set
        num_test_combos = int(
            np.ceil(frac_combinations_test * len(combo_guide_identities))
        )
        test_combos = combo_guide_identities[-num_test_combos:]

        # split combo cells
        # splitting is done before filtering by train / test combos to ensure that samples
        # accumulate across different frac_combinations_train values
        obs_combo = obs[obs["num_guides"] == 2]
        obs_combo_tr_val, obs_combo_test = train_test_split(
            obs_combo,
            test_size=0.2,
            random_state=split_seed,
        )
        obs_combo_tr, obs_combo_val = train_test_split(
            obs_combo_tr_val,
            test_size=0.2,
            random_state=split_seed,
        )
        obs_combo_tr = obs_combo_tr[obs_combo_tr["guide_identity"].isin(train_combos)]
        obs_combo_val = obs_combo_val[
            obs_combo_val["guide_identity"].isin(train_combos)
        ]
        obs_combo_test = obs_combo_test[
            obs_combo_test["guide_identity"].isin(test_combos)
        ]

        # split non-combo cells
        obs_no_combo = obs[obs["num_guides"] < 2]
        obs_no_combo_tr, obs_no_combo_val = train_test_split(
            obs_no_combo, test_size=0.2, random_state=split_seed
        )

        # generate full split obs dataframes
        obs_tr = pd.concat([obs_no_combo_tr, obs_combo_tr])
        obs_val = pd.concat([obs_no_combo_val, obs_combo_val])
        obs_test = obs_combo_test

        split_labels = pd.Series(index=obs.index.copy(), data=None)
        split_labels.loc[obs.index.isin(obs_tr.index)] = "train"
        split_labels.loc[obs.index.isin(obs_val.index)] = "val"
        split_labels.loc[obs.index.isin(obs_test.index)] = "test"

        return split_labels


class NormanDataEfficiencyDataModule(BaseNormanDataModule):
    def __init__(
        self,
        frac_combination_cells_train: float,
        split_seed: int = 0,
        encode_combos_as_unique: bool = False,
        batch_size: int = 128,
        highly_variable_genes_only: bool = False,
        data_path: Optional[str] = None,
    ):
        split_kwargs = dict(
            frac_combination_cells_train=frac_combination_cells_train,
            split_seed=split_seed,
        )
        super().__init__(
            split_kwargs=split_kwargs,
            encode_combos_as_unique=encode_combos_as_unique,
            batch_size=batch_size,
            highly_variable_genes_only=highly_variable_genes_only,
            data_path=data_path,
        )

    @staticmethod
    def _get_split_labels(
        obs: pd.DataFrame,
        split_seed: int,
        # used only by OOD data module
        frac_combinations_train: float = 0,
        frac_combinations_test: float = 0.2,
        # used only by data efficiency data module
        frac_combination_cells_train: float = 0,
    ):
        # TODO: how to implement cleanly? Mypy error for changing signature
        # from superclass

        # split combo cells
        # splitting is done before filtering by train / test combos to ensure that samples
        # accumulate across different frac_combinations_train values
        obs_combo = obs[obs["num_guides"] == 2]
        obs_combo_tr_val, obs_combo_test = train_test_split(
            obs_combo,
            test_size=0.2,
            random_state=split_seed,
        )
        obs_combo_tr, obs_combo_val = train_test_split(
            obs_combo_tr_val,
            test_size=0.2,
            random_state=split_seed,
        )
        # further subsample obs_combo_tr to assess data efficiency
        n_combo_tr_cells = int(frac_combination_cells_train * obs_combo_tr.shape[0])
        obs_combo_tr = obs_combo_tr.iloc[:n_combo_tr_cells]

        # split non-combo cells
        obs_no_combo = obs[obs["num_guides"] < 2]
        obs_no_combo_tr, obs_no_combo_val = train_test_split(
            obs_no_combo, test_size=0.2, random_state=split_seed
        )

        # generate full split obs dataframes
        obs_tr = pd.concat([obs_no_combo_tr, obs_combo_tr])
        obs_val = pd.concat([obs_no_combo_val, obs_combo_val])
        obs_test = obs_combo_test

        split_labels = pd.Series(index=obs.index.copy(), data=None)
        split_labels.loc[obs.index.isin(obs_tr.index)] = "train"
        split_labels.loc[obs.index.isin(obs_val.index)] = "val"
        split_labels.loc[obs.index.isin(obs_test.index)] = "test"

        return split_labels


def get_guide_one_hot_cols(obs: pd.DataFrame):
    guide_one_hot_cols = [
        col
        for col in obs.columns
        if "guide_" in col and col not in ("guide_identity", "guide_ids")
    ]
    return guide_one_hot_cols
