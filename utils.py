#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:13:49 2022

@author: ri21540
"""
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import mean_absolute_error
from training_params import ARGS

def normalise_data(pd_dataframe):
    ARGS.z_norm_dict = read_dict('data/ARGS.z_norm_dict.txt')
    for feature in ARGS.features:
        if feature in ['station', 'isHoliday']:
            continue
        pd_dataframe[feature] = pd_dataframe[feature].apply(lambda x: (x - ARGS.z_norm_dict[feature][0]) / ARGS.z_norm_dict[feature][1])
    return pd_dataframe

def create_max_docks_per_station_dict():
    """ Create dictionary as {station: numDocks}"""
    dir = './data/Train/Train'
    files = os.listdir(dir)
    dfs = []
    max_docks_per_station = dict()
    for file in files:
        if os.path.splitext(file)[-1] == '.csv':
            abs_path = os.path.join(dir, file)
            df = pd.read_csv(abs_path)[['station', 'numDocks']]
            max_docks_per_station[int(df.station.unique().item())] = int(df.numDocks.unique().item())
    return max_docks_per_station

def create_znorm_dict():
    """ Z-norm over all the ARGS.features """
    dir = './data/Train/Train'
    files = os.listdir(dir)
    dfs = []
    ARGS.z_norm_dict = dict()
    for file in files:
        if os.path.splitext(file)[-1] == '.csv':
            abs_path = os.path.join(dir, file)
            df = pd.read_csv(abs_path)
            dfs.append(df)
    dataframe = pd.concat(dfs)    # complete df

    for col in dataframe.columns:
        values = dataframe[col]
        if values.dtype == 'O':    # this checks that the column is of type "object" (string)
            continue
        mean = np.mean(values)
        std_dev = np.std(values)
        ARGS.z_norm_dict[col] = [mean, std_dev]
    return ARGS.z_norm_dict

def test_MAE(X_test, y_test, clfs):
    # MAE - we use this becuase getting close to the true prediction is what we want, not exactly the right bikes like accuracy would give
    # print('Train MAE: ', mean_absolute_error(y_train, clf.predict(X_train)))
    # print('Test MAE: ', mean_absolute_error(y_test, clf.predict(X_test)))

    return mean_absolute_error(y_test, model_predict(clfs, X_test))

def model_predict(clfs, X_test):
    predictions = []
    for clf in clfs:
        predictions.append(clf.predict(X_test))
    prediction = np.mean(predictions, axis=0)
    # If ARGS.target is bikes percent, we need to scale the output
    if ARGS.target == 'bikes_percent':
        station = X_test[:, -1]    # station is the last element in the array.
        numDocks = np.array([MAX_DOCKS_PER_STATION[str(int(i))] for i in station])
        prediction = (numDocks * prediction)
    return prediction

# write validation results to file for comparision of models later
def write_results(results, name, file='./valid/results.csv'):
    if os.path.exists(file):
        data = pd.read_csv(file)
        data[name] = [results]
    else:
        dir = os.path.dirname(file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        data = pd.DataFrame({name: [results]})
    data.to_csv(file, index=False)

def read_results(file='./valid/results.csv'):
    return pd.read_csv(file)

def string_to_list(string):
    # string: '[2.11, 3.11]'
    list_strings = string[1:-1].split(', ')
    list_float = [float(x) for x in list_strings]
    return list_float

def read_dict(path):
    with open(path,'r') as json_file:
        return json.load(json_file)

def write_dict(dic, path):
    with open(path,'w') as fp:
        fp.write(json.dumps(dic))


if __name__=='__main__':
    # Pipeline
    create_numDocks = True
    create_zNorm = True

    if create_numDocks:
        max_docks_per_station = create_max_docks_per_station_dict()
        write_dict(max_docks_per_station, 'data/max_docks_per_station.txt')

    if create_zNorm:
        ARGS.z_norm_dict = create_znorm_dict()
        write_dict(ARGS.z_norm_dict, 'data/ARGS.z_norm_dict.txt')
