#!/bin/bash

# make sure conda is accessable
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# use virtual env
VENV=bikes
conda activate $VENV

# rm valid/results.csv

RUNS=10
MODEL=RandomForest
# MODEL=XGBoost
# MODEL=Linear

python phase1.py -fs -q --val_runs $RUNS -m $MODEL --features 'bikes_3h_ago'
python phase1.py -fs -q --val_runs $RUNS -m $MODEL --features 'short_profile_bikes'
python phase1.py -fs -q --val_runs $RUNS -m $MODEL --features 'short_profile_3h_diff_bikes'
python phase1.py -fs -q --val_runs $RUNS -m $MODEL --features 'isHoliday' 'weekhour' 'day' 'temperature.C'
python phase1.py -fs -q --val_runs $RUNS -m $MODEL --features 'bikes_3h_ago'  'isHoliday' 'weekhour' 'day' 'temperature.C'
python phase1.py -fs -q --val_runs $RUNS -m $MODEL --features 'short_profile_3h_diff_bikes' 'isHoliday' 'weekhour' 'day' 'temperature.C'
python phase1.py -fs -q --val_runs $RUNS -m $MODEL --features 'short_profile_bikes' 'short_profile_3h_diff_bikes' 'isHoliday' 'weekhour' 'day' 'temperature.C'
python phase1.py -fs -q --val_runs $RUNS -m $MODEL --features 'bikes_3h_ago' 'short_profile_bikes' 'short_profile_3h_diff_bikes' 'isHoliday' 'weekhour' 'day' 'temperature.C'

python plot_results.py diff_features_$MODEL
mv valid/results.csv valid/diff_features_single_$MODEL.csv
