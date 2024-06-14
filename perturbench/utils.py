import os 
import pickle
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch


def get_GO_edge_list(args):
    """
    Get gene ontology edge list
    """
    g1, gene2go = args
    edge_list = []
    for g2 in gene2go.keys():
        score = len(gene2go[g1].intersection(gene2go[g2])) / len(
            gene2go[g1].union(gene2go[g2]))
        if score > 0.1:
            edge_list.append((g1, g2, score))
    return edge_list

def make_GO(data_path, pert_list, data_name, num_workers=25, save=True):
    """
    Creates Gene Ontology graph from a custom set of genes
    Adapted from https://github.com/snap-stanford/GEARS/blob/master/gears/utils.py
    Accessed 14/06/2024
    """

    fname = './data/go_essential_' + data_name + '.csv'
    if os.path.exists(fname):
        return pd.read_csv(fname)
    # WE RENAMED gene2go_all TO gene2go.pkl
    with open(os.path.join(data_path, 'gene2go.pkl'), 'rb') as f:
        gene2go = pickle.load(f)
    gene2go = {i: gene2go[i] for i in pert_list if i in gene2go.keys()}

    print('Creating custom GO graph, this can take a few minutes')
    with Pool(num_workers) as p:
        all_edge_list = list(
            tqdm(
                p.imap(
                    get_GO_edge_list, ((g, gene2go) for g in gene2go.keys())
                ),
                total=len(gene2go.keys())
            )
        )
    edge_list = []
    for i in all_edge_list:
        edge_list = edge_list + i

    df_edge_list = pd.DataFrame(edge_list).rename(
        columns={0: 'source', 1: 'target', 2: 'importance'})

    if save:
        print('Saving edge_list to file')
        df_edge_list.to_csv(fname, index=False)

    return df_edge_list


def get_map(pert, gene_list, df):
    '''
    Get gene-gene importance score
    From https://github.com/nitzanlab/biolord_reproducibility/blob/main/'
    'notebooks/perturbations/norman/1_perturbations_norman_preprocessing.ipynb

    '''
    tmp = pd.DataFrame(np.zeros(len(gene_list)), index=gene_list)
    tmp.loc[df[df.target == pert].source.values, :] = df[df.target == pert].importance.values[:, np.newaxis]
    return tmp.values.flatten()

def bool2idx(x):
    """
    Returns the indices of the True-valued entries in a boolean array `x`
    From https://github.com/nitzanlab/biolord_reproducibility/blob/main/utils/utils_perturbations.py
    Accessed 14/06/2024
    """
    return np.where(x)[0]


def repeat_n(x, n):
    """combo_seen2
    Returns an n-times repeated version of the Tensor x,
    repetition dimension is axis 0
    https://github.com/nitzanlab/biolord_reproducibility/
    blob/main/utils/utils_perturbations.py
    """
    # copy tensor to device BEFORE replicating it n times
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device).view(1, -1).repeat(n, 1)