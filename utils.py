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
from training_params import FEATURES, TARGET

def normalise_data(pd_dataframe):
    Z_NORM_dict = read_dict('data/Z_NORM_dict.txt')
    for feature in FEATURES:
        if feature in ['station', 'isHoliday']:
            continue
        pd_dataframe[feature] = pd_dataframe[feature].apply(lambda x: (x - Z_NORM_dict[feature][0]) / Z_NORM_dict[feature][1])
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

def creat_z_norm_dict():
    """ Z-norm over all the FEATURES """
    dir = './data/Train/Train'
    files = os.listdir(dir)
    dfs = []
    Z_NORM_dict = dict()
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
        Z_NORM_dict[col] = [mean, std_dev]
    return Z_NORM_dict

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
    # If TARGET is bikes percent, we need to scale the output
    if TARGET == 'bikes_percent':
        station = X_test[:, -1]    # station is the last element in the array.
        numDocks = np.array([MAX_DOCKS_PER_STATION[str(int(i))] for i in station])
        prediction = (numDocks * prediction)
    return prediction

def read_dict(path):
    with open(path,'r') as json_file:
        return json.load(json_file)

def write_dict(dic, path):
    with open(path,'w') as fp:
        fp.write(json.dumps(dic))


if __name__=='__main__':
    # Pipeline
    create_numDocks = True
    create_Z_NORM = True

    if create_numDocks:
        max_docks_per_station = create_max_docks_per_station_dict()
        write_dict(max_docks_per_station, 'data/max_docks_per_station.txt')

    if create_Z_NORM:
        Z_NORM_dict = creat_z_norm_dict()
        write_dict(Z_NORM_dict, 'data/Z_NORM_dict.txt')
