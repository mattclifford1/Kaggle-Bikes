
#!/bin/bash

# make sure conda is accessable
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# use virtual env
VENV=bikes
conda activate $VENV

rm valid/results.csv

RUNS=10
models=(SVR\ XGBoost\ RandomForest XGBoost\ RandomForest)

for phase2 in 1 2 3 ; do
  python phase_3.py -q --val_runs $RUNS -m SVR XGBoost RandomForest  --num_phase_2_models $phase2
  python phase_3.py -q --val_runs $RUNS -m XGBoost RandomForest  --num_phase_2_models $phase2
done


python plot_results.py phase_3_ensembles
mv valid/results.csv valid/phase_3_ensembles.csv
