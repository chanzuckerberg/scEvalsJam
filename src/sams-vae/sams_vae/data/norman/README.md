## Exploring genetic interaction manifolds constructed from rich single-cell phenotypes

Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6746554/

```
Norman TM, Horlbeck MA, Replogle JM, Ge AY, Xu A, Jost M, Gilbert LA, Weissman JS. Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. Science. 2019 Aug 23;365(6455):786-793. doi: 10.1126/science.aax4438. Epub 2019 Aug 8. PMID: 31395745; PMCID: PMC6746554.
```

Dataset of gene overexpression in single genes and combination using CRISPRa, designed to model gene interactions.

### Data
We use the curated and preprocessed anndata files from Theis lab.
https://github.com/theislab/sc-pert. Specifically, we use the "Norman processed" dataset.

Data download URL: https://ndownloader.figshare.com/files/34027562

Data processing overview:
* Raw data curated from GEO and formatted as anndata ([curation notebook](https://nbviewer.ipython.org/github/theislab/sc-pert/blob/main/datasets/Norman_2019_curation.ipynb))
* Processed data generated with basic QC ([processing notebook](https://nbviewer.ipython.org/github/theislab/sc-pert/blob/main/datasets/Norman_2019.ipynb))
    * Index genes by symbol
    * Filter genes with fewer than 200 genes
    * Filter genes expressed by fewer than 20 cells
    * Filter cells with >15% counts expressed by mitochondrial genes
    * Filter cells with > 6000 genes expressed(?)
    * Normalize counts per cell (express as fraction of counts, raw counts saved as layer)
    * Apply log to normalized counts
    * Add perturbation names + gene symbols
    * Precompute PCs / UMAP
