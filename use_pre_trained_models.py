import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
from tqdm import tqdm
from utils import write_dict, read_dict
from read_data import get_all_model_csvs
import matplotlib.pyplot as plt

dict_path = './data/linear_models_MAEs.txt'
load_model = True

def pred_models(model_csvs, test_df):
    '''
    predict and average multiple models
    '''
    preds = []
    for model_csv in model_csvs:
        pred = pred_pretrained_model(model_csv, test_df)
        preds.append(pred_pretrained_model(model_csv, test_df))
    return np.mean(np.array(preds), axis=0)  # take mean of ensemble


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

def MAE(preds, test_df):
    return mean_absolute_error(test_df['bikes'].to_numpy(), preds)

def run_all_stations(num_model_to_use=2, dir='./data/Train/Train'):
    '''
    get dict of MAE for each model at each station
    '''
    # load models is already computed
    if os.path.exists(dict_path) and load_model:
        MAEs_dict = read_dict(dict_path)
    else:
        model_csvs = get_all_model_csvs()
        # go through all the training data
        train_stations = os.listdir(dir)
        MAEs_dict = {}
        for station in tqdm(train_stations):
            if os.path.splitext(station)[-1] == '.csv':
                train_df = get_train_data_with_intercept(station, dir)
                # now find which pretrained model 'fits' this station the best
                # by getting MAE for all linear models
                MAEs = []
                for model_csv in model_csvs:
                    MAEs.append(MAE_model(model_csv, train_df))
                # save MAEs to dict
                MAEs_dict[station] = MAEs
        write_dict(MAEs_dict, dict_path)
    return MAEs_dict

def get_train_data_with_intercept(station, dir='./data/Train/Train'):
    # read train data for current station
    abs_path = os.path.join(dir, station)
    train_df = pd.read_csv(abs_path).dropna().reset_index(drop=True)
    # add intercept for matrix dot product linear models
    train_df['(Intercept)'] = np.ones(len(train_df))
    return train_df

def get_top_models(MAEs_dict, num_model_to_use=2):
    best_model_each_station = {}
    best_model_error = []
    for station in MAEs_dict.keys():
        model_csvs = get_all_model_csvs()
        MAEs = MAEs_dict[station]
        # take the i best models
        best_model_each_station[station] = []
        k_args_mins = np.argpartition(MAEs, kth=num_model_to_use)[:num_model_to_use]
        for arg in k_args_mins:
            best_model_each_station[station].append(model_csvs[arg])
        # read train data for current station
        train_df = get_train_data_with_intercept(station)
        # predict all together
        pred_together = pred_models(best_model_each_station[station], train_df)
        best_model_error.append(MAE(pred_together, train_df))
    return best_model_each_station, np.mean(best_model_error)


def print_top_models(dir='./data/Train/Train'):
    MAEs_dict = run_all_stations(dir)
    best_model_each_station, mean_all = get_top_models(MAEs_dict, num_model_to_use=1)
    # analyse what models are selected/ performing best
    results_dict = {}
    for model in best_model_each_station.values():
        model_name = '-'.join(model.split('_')[4:])
        if model_name in results_dict.keys():
            results_dict[model_name] += 1
        else:
            results_dict[model_name] = 0
    print(results_dict)

def compare_num_models(dir='./data/Train/Train'):
    MAEs = []
    MAEs_dict = run_all_stations(dir)
    num_models = []
    for i in tqdm(range(1, 200)):
        _, mean_all = get_top_models(MAEs_dict, num_model_to_use=i)
        MAEs.append(mean_all)
        num_models.append(i)
    print(MAEs)
    plt.plot(num_models, MAEs)
    plt.xlabel('Number of linear models in ensemble')
    plt.ylabel('average MAE')
    plt.show()


if __name__ == '__main__':
    compare_num_models()


    # dict_path = './data/linear_models_MAEs.txt'
    # load_model = False
    # if os.path.exists(dict_path) and load_model:
    #     best_model_each_station = read_dict(dict_path)
    # else:
    #     best_model_each_station, mean_all = run_all_stations()
    #     write_dict(best_model_each_station, dict_path)
    #     print(mean_all)
