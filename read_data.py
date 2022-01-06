import pandas as pd
import os
from utils import normalise_data
from sklearn.model_selection import train_test_split
from training_params import ARGS

def get_csv_file_from_num(num):
    return './data/Train/Train/station_'+str(num)+'_deploy.csv'

def split_training_data(csv_file, test_size=0.33):
    pd_dataframe = pd.read_csv(csv_file)
    # convert weekdays to ints
    days = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    pd_dataframe['weekday'] = [days[x] for x in pd_dataframe['weekday']]
    #get rid of na rows
    pd_dataframe= pd_dataframe.dropna()
    pd_dataframe['bikes_percent'] = (pd_dataframe['bikes'] / pd_dataframe['numDocks'])
    if ARGS.z_norm:
        pd_dataframe = normalise_data(pd_dataframe)
    # split into train/val
    X_train, X_test, y_train, y_test = train_test_split(pd_dataframe[ARGS.features].to_numpy(),
                                                        pd_dataframe['bikes'].to_numpy(),
                                                        test_size=test_size)
                                                        #random_state=42)
    return X_train, X_test, y_train, y_test

def get_training_csvs(dir='./data/Train/Train'):
    training_csvs = []
    files = os.listdir(dir)
    for file in files:
        if os.path.splitext(file)[-1] == '.csv':
            abs_path = os.path.join(dir, file)
            training_csvs.append(abs_path)
    return training_csvs


def get_all_model_csvs(dir='./data/Models/Models'):
    csvs = []
    files = os.listdir(dir)
    for file in files:
        if os.path.splitext(file)[-1] == '.csv':
            csvs.append(os.path.join(dir, file))
    return csvs


#
# if __name__ == '__main__':
