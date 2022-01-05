import pandas as pd
import os
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import xgboost as xg
from utils import read_dict

# GLOBAL VARIABLES
target = 'bikes'    # 'bikes' or 'bikes_percent'
models_list = ['SVR']      # it must be a list - ['SVR', 'XGBoost']
numDocks_dict = read_dict('data/numDocks_dict.txt')
z_norm = True
save_output = False

def get_csv_file_from_num(num):
    return './data/Train/Train/station_'+str(num)+'_deploy.csv'

def read_training_data(csv_file, test_size=0.33):
    pd_dataframe = pd.read_csv(csv_file)
    # convert weekdays to ints
    days = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    pd_dataframe['weekday'] = [days[x] for x in pd_dataframe['weekday']]
    #get rid of na rows
    pd_dataframe= pd_dataframe.dropna()
    pd_dataframe['bikes_percent'] = (pd_dataframe['bikes'] / pd_dataframe['numDocks'])
    # what features to use
    features = ['bikes_3h_ago',
                'short_profile_bikes',
                'short_profile_3h_diff_bikes',
                'isHoliday',
                'weekhour',
                'day',
                'station']   # this MUST be the last feature
    if z_norm:
        z_norm_dict = read_dict('data/z_norm_dict.txt')
        for feature in features:
            if feature in ['station', 'isHoliday']:
                continue
            pd_dataframe[feature] = pd_dataframe[feature].apply(lambda x: (x - z_norm_dict[feature][0]) / z_norm_dict[feature][1])
    # split into train/val
    X_train, X_test, y_train, y_test = train_test_split(pd_dataframe[features].to_numpy(),
                                                        pd_dataframe['bikes'].to_numpy(),
                                                        test_size=test_size)
                                                        #random_state=42)
    return X_train, X_test, y_train, y_test

def train(X_train, y_train):
    """ Train a model and return it"""
    clfs =[]
    if target == 'bikes_percent':
        y_train = y_train  / [numDocks_dict[str(int(X_train[i, -1]))] for i in range(X_train.shape[0])]
    if 'SVR' in models_list:
        # fit SVM
        clf = make_pipeline(StandardScaler(), SVR(gamma='auto'))
        clf.fit(X_train, y_train)
        clfs.append(clf)
    if 'XGBoost' in models_list:
        clf = xg.XGBRegressor(objective ='reg:squarederror',
              n_estimators = 20, seed = 123)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    return clfs

def model_predict(clfs, X_test):
    predictions = []
    for clf in clfs:
        predictions.append(clf.predict(X_test))
    prediction = np.mean(predictions, axis=0)
    # If target is bikes percent, we need to scale the output
    if target == 'bikes_percent':
        station = X_test[:, -1]    # station is the last element in the array.
        numDocks = np.array([numDocks_dict[str(int(i))] for i in station])
        prediction = (numDocks * prediction)
    return prediction
    
def test_MAE(X_test, y_test, clfs):
    # MAE - we use this becuase getting close to the true prediction is what we want, not exactly the right bikes like accuracy would give
    # print('Train MAE: ', mean_absolute_error(y_train, clf.predict(X_train)))
    # print('Test MAE: ', mean_absolute_error(y_test, clf.predict(X_test)))
    
    return mean_absolute_error(y_test, model_predict(clfs, X_test))

def run_single_and_all_stations():
    """Train and test on single stations and on all stations.
    Return the MAE for both cases.
    """
    dir = './data/Train/Train'
    files = os.listdir(dir)
    results = []
    dataframes = {'X_train': [],
                  'X_test': [],
                  'y_train': [],
                  'y_test': []}
    for file in files:
        if os.path.splitext(file)[-1] == '.csv':
            abs_path = os.path.join(dir, file)
            X_train, X_test, y_train, y_test  = read_training_data(abs_path)
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


if __name__ == '__main__':
    # Validate on single and all stations
    iterate_all()
    
    pd_dataframe = pd.read_csv('./data/test.csv')
    y_preds = []
    num_prev = -1   # there are no -1 value station
    for id in tqdm(pd_dataframe['Id']):
        num = pd_dataframe.loc[id-1, 'station']
        if num != num_prev:
            X_train, _, y_train, _ = read_training_data(get_csv_file_from_num(num), test_size=int(1))
            clfs = train(X_train, y_train)
            features = ['bikes_3h_ago',
                        'short_profile_bikes',
                        'short_profile_3h_diff_bikes',
                        'isHoliday',
                        'weekhour',
                        'day',
                        'station']   # station needs to be the last element
        X_test = np.expand_dims(pd_dataframe.loc[id-1][features].values, 0)
        y_pred = model_predict(clfs, X_test)
        y_preds.append([id, int(y_pred[0])])
        num_prev = num
        # if id == 5:
        #     break
    df = pd.DataFrame(y_preds)
    if save_output:
        df.to_csv('preds_XGBoost.csv', index=False, header=['Id', 'bikes'])
