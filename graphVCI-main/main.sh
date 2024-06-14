#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate gvci-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name train_epoch-1000"
PYARGS="$PYARGS --artifact_path $DATA/artifact"
PYARGS="$PYARGS --data_path $DATA/datasets/marson_prepped.h5ad" #sciplex_prepped.h5ad, L008_prepped.h5ad
PYARGS="$PYARGS --graph_path $DATA/graphs/marson_grn_128.pth" #sciplex_grn_128.pth, L008_grn_128.pth
PYARGS="$PYARGS --gpu 0" #PYARGS="$PYARGS --cpu"

PYARGS="$PYARGS --omega1 50.0"
PYARGS="$PYARGS --outcome_dist normal"
PYARGS="$PYARGS --dist_mode match"
PYARGS="$PYARGS --decode_aggr dot"

PYARGS="$PYARGS --max_epochs 1000"
PYARGS="$PYARGS --batch_size 64"
PYARGS="$PYARGS --eval_mode classic"

python main.py $PYARGS
