import scipy as sp
import pandas as pd
import anndata as ad


class PerturbationDataset:
    """Class responsible for providing perturbation data for model training and prediction, as well as
    harmonising the attributes available in a dataset. Uses AnnData."""

    def __init__(self, anndata: ad.AnnData, perturbation_field, covariate_fields, name: str, description: str) -> None:
        self.raw_counts = anndata.X
        self.perturbations = anndata.obs[perturbation_field]
        self.covariates = anndata.obs[covariate_fields]
        self.name = name
        self.description = description

    def get_raw_counts(self) -> sp.sparse.csr_matrix:
        return self.raw_counts

    def get_covariates(self) -> pd.DataFrame:
        return self.covariates

    def get_perturbations(self) -> pd.Series:
        return self.perturbations

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def anndata(self) -> ad.AnnData:
        anndata = ad.AnnData(self.raw_counts, obs=self.covariates)
        anndata.obs["perturbation"] = self.perturbations
        return anndata
