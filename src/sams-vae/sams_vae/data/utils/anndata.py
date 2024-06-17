import anndata
import numpy as np
import pandas as pd


def align_adatas(*adatas: anndata.AnnData):
    dfidxs = [
        pd.DataFrame({_i: np.arange(len(_adata)), "key": _adata.obs.index.values})
        for _i, _adata in enumerate(adatas)
    ]
    dfidx = dfidxs[0]
    for i in range(1, len(dfidxs)):
        dfidx = dfidx.merge(dfidxs[i], on="key", how="inner")
    return [_adata[dfidx[_i].values] for _i, _adata in enumerate(adatas)]
