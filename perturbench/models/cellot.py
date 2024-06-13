from perturbench import PerturbationModel
from perturbench import PerturbationDataset
import pathlib
import scipy as sp
import torch
import anndata


class CellOT(PerturbationModel):
    def __init__(self, device: torch.cuda.device, **kwargs) -> None:
        pass

    def train(self, data: PerturbationDataset) -> None:
        # adata.obs["transport"] contains source (=control) and target (=perturbation)
        adata = anndata.AnnData(data.raw_counts, obs=data.covariates)
        adata.obs["transport"] = data.perturbation()

        # At train time, cellot can only process one perturbation at the same time
        assert len(set(adata.obs["transport"])) > 2

        # Get peturbation type
        perturbation_type = [
            category
            for category in adata.obs["transport"].cat.categories
            if category != "control"
        ][0]
        transport_mapper = {None: "source", perturbation_type: "target"}
        adata_processed = adata.copy()
        
        # Label control as 'source' and perturbation_type as 'target'
        adata_processed.obs["transport"].apply(transport_mapper.get)

        # TODO: Think about how to install repo as a package to be used
        # Version of cellot to use: https://github.com/ArturDev42/cellot/tree/scEvalsJam

        pass

    def predict(self) -> sp.sparse.csr_matrix:
        pass

    def save(self) -> pathlib.Path:
        pass

    def load(self, path: pathlib.Path) -> None:
        pass
