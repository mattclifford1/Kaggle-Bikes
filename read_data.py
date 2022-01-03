import pandas as pd
import os
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_csv_file_from_num(num):
    return './data/Train/Train/station_'+str(num)+'_deploy.csv'

def read_csv(csv_file, test_size=0.33):
    pd_dataframe = pd.read_csv(csv_file)
    # convert weekdays to ints
    days = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    pd_dataframe['weekday'] = [days[x] for x in pd_dataframe['weekday']]
    #get rid of na rows
    pd_dataframe= pd_dataframe.dropna()
    # what features to use
    features = ['bikes_3h_ago',
                'short_profile_bikes',
                'short_profile_3h_diff_bikes',
                'isHoliday',
                'weekhour',
                'day']
    target = 'bikes'
    # split into train/val
    X_train, X_test, y_train, y_test = train_test_split(pd_dataframe[features].to_numpy(),
                                                        pd_dataframe[target].to_numpy(),
                                                        test_size=test_size)
                                                        # random_state=42)
    return X_train, X_test, y_train, y_test

def train(X_train, y_train):
    # fit SVM
    clf = make_pipeline(StandardScaler(), SVR(gamma='auto'))
    clf.fit(X_train, y_train)
    return clf

def test_MAE(X_test, y_test, clf):
    # MAE - we use this becuase getting close to the true prediction is what we want, not exactly the right bikes like accuracy would give
    # print('Train MAE: ', mean_absolute_error(y_train, clf.predict(X_train)))
    # print('Test MAE: ', mean_absolute_error(y_test, clf.predict(X_test)))
    return mean_absolute_error(y_test, clf.predict(X_test))

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
            X_train, X_test, y_train, y_test  = read_csv(abs_path)
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
    
    save_output=False
    pd_dataframe = pd.read_csv('./data/test.csv')
    y_preds = []
    num_prev = -1   # there are no -1 value station
    for id in tqdm(pd_dataframe['Id']):
        num = pd_dataframe.loc[id-1, 'station']
        if num != num_prev:
            X_train, _, y_train, _ = read_csv(get_csv_file_from_num(num), test_size=int(1))
            clf = train(X_train, y_train)
            features = ['bikes_3h_ago',
                        'short_profile_bikes',
                        'short_profile_3h_diff_bikes',
                        'isHoliday',
                        'weekhour',
                        'day']
        X_test = np.expand_dims(pd_dataframe.loc[id-1][features].values, 0)
        y_pred = clf.predict(X_test)
        y_preds.append([id, int(y_pred[0])])
        num_prev = num
        # if id == 5:
        #     break
    df = pd.DataFrame(y_preds)
    if save_output:
        df.to_csv('preds.csv', index=False, header=['Id', 'bikes'])
