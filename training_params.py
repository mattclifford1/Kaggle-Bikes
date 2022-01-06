# GLOBAL VARIABLES
TARGET = 'bikes'    # 'bikes' or 'bikes_percent'
MODELS_LIST = ['XGBoost']      # it must be a list - ['SVR', 'XGBoost']
MAX_DOCKS_PER_STATION_FILE = 'data/max_docks_per_station.txt'
Z_NORM = True
QUICK_VALIDATION = True
SAVE_TEST_PREDS = True
# what FEATURES to use
FEATURES = ['bikes_3h_ago',
            'short_profile_bikes',
            'short_profile_3h_diff_bikes',
            'isHoliday',
            'weekhour',
            'day']
            # ,
            # 'station']   # this MUST be the last feature
