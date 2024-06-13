import scipy as sp
import pandas as pd
import anndata as ad


class PerturbationDataset:
    """Class responsible for providing perturbation data for model training and prediction."""

    def __init__(self, anndata: ad.AnnData) -> None:
        pass

    def raw_counts(self) -> sp.sparse.csr_matrix:
        pass

    def covariates(self) -> pd.DataFrame:
        pass

    def perturbation(self) -> pd.Series:
        """

        :return:
        y: pd.Series
            Returns a pd.Series where None denotes the control condition, genetic perturbations
            are encoded as "GENETIC:HUGO_SYMBOL" and small molecule perturbations are encoded as "CHEMICAL:CHEMBL".
        """
        pass
