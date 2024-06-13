import anndata


class PerturbationEval:
    """Class responsible for creating Anndata object for the evalulation."""

    def __init__(self, pred, ground_truth):
        self.eval_adata = self.create_eval_adata(pred, ground_truth)
        
    def create_eval_adata(self, pred, ground_truth):
        adata = anndata.AnnData(X=pred, obs=ground_truth)
        return adata