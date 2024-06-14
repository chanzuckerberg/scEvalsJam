import scipy as sp
import pandas as pd
import anndata as ad


class PerturbationDataset:
    """Class responsible for providing perturbation data for model training and prediction, as well as
    harmonising the attributes available in a dataset. Uses AnnData."""

    def __init__(self, anndata: ad.AnnData, perturbation_field, covariate_fields) -> None:
        self.raw_counts = anndata.X
        self.perturbations = anndata.obs[perturbation_field]
        self.covariates = anndata.obs[covariate_fields]

    def raw_counts(self) -> sp.sparse.csr_matrix:
        return self.raw_counts

    def covariates(self) -> pd.DataFrame:
        return self.covariates

    def perturbation(self) -> pd.Series:
        return self.perturbations

    def anndata(self) -> ad.AnnData:
        anndata = ad.AnnData(self.raw_counts, obs=self.covariates)
        anndata.obs["perturbation"] = self.perturbations
        return anndata
