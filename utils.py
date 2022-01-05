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

def create_numDocks_Dict():
    """ Create dictionary as {station: numDocks}"""
    dir = './data/Train/Train'
    files = os.listdir(dir)
    dfs = []
    numDocks_dict = dict()
    for file in files:
        if os.path.splitext(file)[-1] == '.csv':
            abs_path = os.path.join(dir, file)
            df = pd.read_csv(abs_path)[['station', 'numDocks']]
            numDocks_dict[int(df.station.unique().item())] = int(df.numDocks.unique().item())
    return numDocks_dict

def z_norm():
    """ Z-norm over all the features """
    dir = './data/Train/Train'
    files = os.listdir(dir)
    dfs = []
    z_norm_dict = dict()
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
        z_norm_dict[col] = [mean, std_dev]
    return z_norm_dict


def read_dict(path):
    with open(path,'r') as json_file:
        return json.load(json_file)

if __name__=='__main__':
    # Pipeline
    create_numDocks = False
    create_z_norm = True
    
    if create_numDocks:
        numDocks_dict = create_numDocks_Dict()
        with open('data/numDocks_dict.txt','w') as fp:
            fp.write(json.dumps(numDocks_dict))
    
    if create_z_norm:
        z_norm_dict = z_norm()
        with open('data/z_norm_dict.txt','w') as fp:
            fp.write(json.dumps(z_norm_dict))