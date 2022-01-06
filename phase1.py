import os
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm
import xgboost as xg
from utils import read_dict, test_MAE, model_predict, normalise_data
from read_data import get_csv_file_from_num, split_training_data, get_training_csvs
from training_params import TARGET, MODELS_LIST, MAX_DOCKS_PER_STATION_FILE, Z_NORM, QUICK_VALIDATION, SAVE_TEST_PREDS, FEATURES

MAX_DOCKS_PER_STATION = read_dict(MAX_DOCKS_PER_STATION_FILE)

def train(X_train, y_train):
    """ Train a model and return it"""
    clfs =[]
    if TARGET == 'bikes_percent':
        y_train = y_train  / [MAX_DOCKS_PER_STATION[str(int(X_train[i, -1]))] for i in range(X_train.shape[0])]
    if 'SVR' in MODELS_LIST:
        # fit SVM
        clf = make_pipeline(StandardScaler(), SVR(gamma='auto'))
        clf.fit(X_train, y_train)
        clfs.append(clf)
    if 'XGBoost' in MODELS_LIST:
        clf = xg.XGBRegressor(objective ='reg:squarederror',
              n_estimators = 20, seed = 123)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    return clfs


def run_single_and_all_stations():
    """Train and test on single stations and on all stations.
    Return the MAE for both cases.
    """
    results = []
    dataframes = {'X_train': [],
                  'X_test': [],
                  'y_train': [],
                  'y_test': []}
    for training_csv in get_training_csvs():
        X_train, X_test, y_train, y_test  = split_training_data(training_csv)
        dataframes['X_train'].append(X_train)
        dataframes['X_test'].append(X_test)
        dataframes['y_train'].append(y_train)
        dataframes['y_test'].append(y_test)
        results.append(test_MAE(X_test, y_test, train(X_train, y_train)))
    single = sum(results)/len(results)
    # now do all stations at once
    X_trains = np.concatenate(dataframes['X_train'])
    X_tests = np.concatenate(dataframes['X_test'])
    y_trains = np.concatenate(dataframes['y_train'])
    y_tests = np.concatenate(dataframes['y_test'])
    all = test_MAE(X_tests, y_tests, train(X_trains, y_trains))
    return single, all

def iterate_all():
    results_single = []
    results_all = []
    for i in range(5):
        single, all = run_single_and_all_stations()
        results_single.append(single)
        results_all.append(all)
        print(single, all)
    print('========= ALL =========')
    print(f'Results on single stations: {sum(results_single)/len(results_single)}')
    print(f'Results on all stations: {sum(results_all)/len(results_all)}')

def run_test_preds(dir='./data/test.csv'):
    pd_dataframe = pd.read_csv(dir)
    if Z_NORM:
        pd_dataframe = normalise_data(pd_dataframe)
    y_preds = []
    num_prev = -1   # there are no -1 value station
    for id in tqdm(pd_dataframe['Id']):
        num = pd_dataframe.loc[id-1, 'station']
        if num != num_prev:
            X_train, _, y_train, _ = split_training_data(get_csv_file_from_num(num),
                                                         test_size=int(1))
            clfs = train(X_train, y_train)
        X_test = np.expand_dims(pd_dataframe.loc[id-1][FEATURES].values, 0)
        y_pred = model_predict(clfs, X_test)
        y_preds.append([id, int(y_pred[0])])
        num_prev = num
        # if id == 5:
        #     break
    return y_preds

if __name__ == '__main__':
    # Validate on single and all stations
    if QUICK_VALIDATION:
        iterate_all()
    # get predictions for kaggle test data and save to csv for submission
    if SAVE_TEST_PREDS:
        y_preds = run_test_preds()
        df = pd.DataFrame(y_preds)
        df.to_csv('./predictions/preds_z'+str(Z_NORM)+'_models'+str(MODELS_LIST)+str(TARGET)+'.csv', index=False, header=['Id', 'bikes'])
