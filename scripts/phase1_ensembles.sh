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
python phase1.py -q --val_runs $RUNS -m RandomForest XGBoost
python phase1.py -q --val_runs $RUNS -m RandomForest SVR
python phase1.py -q --val_runs $RUNS -m SVR XGBoost
python phase1.py -q --val_runs $RUNS -m RandomForest XGBoost SVR

python plot_results.py ensemble_models_phase1
mv valid/results.csv valid/ensemble_models_phase1.csv
