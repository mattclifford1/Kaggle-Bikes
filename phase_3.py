from phase_2 import get_top_models, get_MAE_all_stations_all_models
from utils import write_dict, read_dict
from utils import read_dict, normalise_data, write_results
from read_data import get_csv_file_from_num, split_training_data_phase3, get_training_csvs
from training_params import ARGS
from phase1 import train
from tqdm import tqdm
import os
from phase_2 import pred_pretrained_model
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

dict_path = './data/linear_models_MAEs.txt'
load_model = True

def test_MAE_phase3(X_test, X_test_df, y_test, clfs):
    # MAE - we use this becuase getting close to the true prediction is what we want, not exactly the right bikes like accuracy would give
    # print('Train MAE: ', mean_absolute_error(y_train, clf.predict(X_train)))
    # print('Test MAE: ', mean_absolute_error(y_test, clf.predict(X_test)))

    return mean_absolute_error(y_test, model_predict_both_phases(clfs, X_test, X_test_df))

def model_predict_both_phases(clfs, X_test, X_test_df):
    predictions = []
    for clf in clfs:
        if type(clf) == str: # pretrained linear model
            preds = pred_pretrained_model(clf, X_test_df)
            flat = np.squeeze(preds)
            predictions.append(flat)
        else: # sklearn model
            predictions.append(clf.predict(X_test))
            x=1
    prediction = np.mean(predictions, axis=0)
    # If ARGS.target is bikes percent, we need to scale the output
    if ARGS.target == 'bikes_percent':
        station = X_test[:, -1]    # station is the last element in the array.
        numDocks = np.array([MAX_DOCKS_PER_STATION[str(int(i))] for i in station])
        prediction = (numDocks * prediction)
    return prediction


def run_single_and_all_stations(phase_2_clfs):
    """Train and test on single stations and on all stations.
    Return the MAE for both cases.
    """
    results = []
    dataframes = {'X_train': [],
                  'X_test': [],
                  'y_train': [],
                  'y_test': []}
    for training_csv in get_training_csvs():
        X_train, X_test, y_train, y_test, X_test_phase2_df  = split_training_data_phase3(training_csv)
        dataframes['X_train'].append(X_train)
        dataframes['X_test'].append(X_test)
        dataframes['y_train'].append(y_train)
        dataframes['y_test'].append(y_test)
        phase_1_clfs = train(X_train, y_train)
        # get the classifiers from phase 2 for this station (key is the filename)
        phase_2_clf = phase_2_clfs[os.path.basename(training_csv)]
        phase_3_clfs = phase_1_clfs + phase_2_clf
        results.append(test_MAE_phase3(X_test, X_test_phase2_df, y_test, phase_3_clfs))
    single = sum(results)/len(results)
    # now do all stations at once
    # X_trains = np.concatenate(dataframes['X_train'])
    # X_tests = np.concatenate(dataframes['X_test'])
    # y_trains = np.concatenate(dataframes['y_train'])
    # y_tests = np.concatenate(dataframes['y_test'])
    # phase_1_clfs = train(X_trains, y_trains)
    # phase_3_clfs = phase_1_clfs + phase_2_clfs
    # all = test_MAE(X_tests, y_tests, phase_3_clfs)
    return single, all

def run_test_preds(phase_2_clfs, dir='./data/test.csv'):
    pd_dataframe = pd.read_csv(dir)
    if ARGS.z_norm:
        pd_dataframe = normalise_data(pd_dataframe)
    y_preds = []
    num_prev = -1   # there are no -1 value station
    for id in tqdm(pd_dataframe['Id']):
        num = pd_dataframe.loc[id-1, 'station']
        if num != num_prev:
            X_train, _, y_train, _, X_test_phase2_df  = split_training_data_phase3(get_csv_file_from_num(num),
                                                         test_size=int(1))
            phase_1_clfs = train(X_train, y_train)
        X_test = np.expand_dims(pd_dataframe.loc[id-1][ARGS.features].values, 0)
        phase_2_clf = phase_2_clfs['station_'+str(num)+'_deploy.csv']
        phase_3_clfs = phase_1_clfs + phase_2_clf
        y_pred = model_predict_both_phases(phase_3_clfs, X_test, X_test_phase2_df)
        y_preds.append([id, int(y_pred[0])])
        num_prev = num
        # if id == 5:
        #     break
    return y_preds

if __name__ == '__main__':
    MAEs_dict = get_MAE_all_stations_all_models()
    phase_2_clfs, _ = get_top_models(MAEs_dict, num_model_to_use=ARGS.num_phase_2_models)

    run_name = str(ARGS.num_phase_2_models)+'-pretrained-models_'+'_'
    for x in ARGS.models_list:
        run_name += str(x)+'_'
    if ARGS.quick_validation:
        # test on val
        results_single = []
        results_all = []
        for i in tqdm(range(ARGS.val_runs)):
            single, all = run_single_and_all_stations(phase_2_clfs)
            results_single.append(single)
            results_all.append(all)
        print('========= ALL =========')
        print(f'Results on single stations: {sum(results_single)/len(results_single)}')

        print(run_name)
        write_results(results_single, run_name+'-single')
        # write_results(results_all, run_name+'-all')

    # get predictions for kaggle test data and save to csv for submission
    if ARGS.save_test_preds:
        y_preds = run_test_preds(phase_2_clfs)
        df = pd.DataFrame(y_preds)
        df.to_csv('./predictions/phase_3_preds_'+run_name+'.csv', index=False, header=['Id', 'bikes'])
