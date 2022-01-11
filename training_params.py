from argparse import ArgumentParser
allowed_models = ['SVR', 'XGBoost', 'Ridge', 'RidgeCV', 'KRidge', 'Linear', 'RandomForest']
parser = ArgumentParser()
parser.add_argument("-m", "--models_list", nargs="+", default=allowed_models)
parser.add_argument("--target", default='bikes')
parser.add_argument("--max_docks_per_station_file", default='data/max_docks_per_station.txt')
parser.add_argument("--z_norm", default=False, action='store_true')
parser.add_argument("-q", "--quick_validation", default=False, action='store_true')
parser.add_argument("--val_runs", default=2, type=int)
parser.add_argument("-s", "--save_test_preds", default=False, action='store_true')
parser.add_argument("-e", "--fit_ensemble", default=False, action='store_true')
parser.add_argument("-fs", "--features_save", default=False, action='store_true') # save features names to database for plot
parser.add_argument("--features", nargs="+", default=['bikes_3h_ago',
                                                     'short_profile_bikes',
                                                     'short_profile_3h_diff_bikes',
                                                     'isHoliday',
                                                     'weekhour',
                                                     'day',
                                                     'temperature.C',
                                                     # 'station'   # this MUST be the last feature
                                                     ])


ARGS = parser.parse_args()

for model in ARGS.models_list:
    if model not in allowed_models:
        raise ValueError('model '+model+' not in available models: '+str(allowed_models))
# # GLOBAL VARIABLES
# ARGS.target = 'bikes'    # 'bikes' or 'bikes_percent'
# ARGS.models_list = ['XGBoost']      # it must be a list - ['SVR', 'XGBoost']
# ARGS.max_docks_per_station_file = 'data/max_docks_per_station.txt'
# ARGS.z_norm = True
# ARGS.quick_validation = True
# ARGS.save_test_preds = True
# # what ARGS.features to use
# ARGS.features = ['bikes_3h_ago',
#             'short_profile_bikes',
#             'short_profile_3h_diff_bikes',
#             'isHoliday',
#             'weekhour',
#             'day']
#             # ,
            # 'station']   # this MUST be the last feature
