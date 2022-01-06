import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
from tqdm import tqdm
from utils import write_dict, read_dict
from read_data import get_all_model_csvs

def pred_pretrained_model(model_csv, test_df):
    model_df = pd.read_csv(model_csv)
    preds = []
    # make prediction
    preds = np.dot(test_df[list(model_df['feature'].values)].to_numpy(),
                   np.expand_dims(model_df['weight'].to_numpy(), axis=1))
    return preds

def MAE_model(model_csv, test_df):
    preds = pred_pretrained_model(model_csv, test_df)
    return mean_absolute_error(test_df['bikes'].to_numpy(), preds)


def run_all_stations(dir='./data/Train/Train'):
    model_csvs = get_all_model_csvs()
    # go through all the training data
    files = os.listdir(dir)
    best_model_each_station = {}
    best_model_error = []
    for file in tqdm(files):
        if os.path.splitext(file)[-1] == '.csv':
            abs_path = os.path.join(dir, file)
            train_df = pd.read_csv(abs_path).dropna().reset_index(drop=True)
            # now find which pretrained model fits this station the best
            MAEs = []
            train_df['(Intercept)'] = np.ones(len(train_df))
            for model_csv in model_csvs:
                MAEs.append(MAE_model(model_csv, train_df))
            argmin = np.argmin(MAEs)
            best_model_each_station[file] = model_csvs[argmin]
            best_model_error.append(np.min(MAEs))
    return best_model_each_station, np.mean(best_model_error)


if __name__ == '__main__':
    dict_path = './data/best_linear_models.txt'
    if os.path.exists(dict_path):
        best_model_each_station = read_dict(dict_path)
    else:
        best_model_each_station, mean_all = run_all_stations()
        write_dict(best_model_each_station, dict_path)
        print(mean_all)
    # analyse what models are selected/ performing best
    results_dict = {}
    for model in best_model_each_station.values():
        model_ARGS.features_name = '-'.join(model.split('_')[4:])
        if model_ARGS.features_name in results_dict.keys():
            results_dict[model_ARGS.features_name] += 1
        else:
            results_dict[model_ARGS.features_name] = 0
    print(results_dict)
