#!/bin/bash

# make sure conda is accessable
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# use virtual env
VENV=bikes
conda activate $VENV

rm valid/results.csv

RUNS=10

# make sure database exists with this run
python phase1.py -q --val_runs $RUNS -m Linear
# run in parralel (requires 55GB RAM) - delete the & for sequencial
for model in SVR XGBoost Ridge KRidge RidgeCV RandomForest; do
  python phase1.py -q --val_runs $RUNS -m "$model"
done

#
# python phase1.py -q --val_runs $RUNS -m Linear
# python phase1.py -q --val_runs $RUNS -m SVR
# python phase1.py -q --val_runs $RUNS -m XGBoost
# python phase1.py -q --val_runs $RUNS -m Ridge
# python phase1.py -q --val_runs $RUNS -m KRidge
# python phase1.py -q --val_runs $RUNS -m RidgeCV
# python phase1.py -q --val_runs $RUNS -m RandomForest

python plot_results.py single_models
mv valid/results.csv valid/single_models.csv
