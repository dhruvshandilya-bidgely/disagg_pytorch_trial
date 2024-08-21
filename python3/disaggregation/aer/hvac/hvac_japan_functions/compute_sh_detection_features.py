"""
Author - Kris Duszak
Date - 3/25/2019
Module to compute t-statistics for extreme temperature energy use vs mid temperature energy use at hourly level
"""

# Import python packages

import logging
import numpy as np
import pandas as pd

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def compute_features_for_sh_detection(hvac_input, hvac_params, vacation_monthly, logger_pass, debug_detection):

    """
    Parameters:
        hvac_input              (np.ndarray)        : Feature matrix for hvac module
        hvac_params             (dict)              : Contains config parameters for hvac
        vacation_monthly        (np.ndarray)        : Aggregated vacation at monthly level
        logger_pass             (dict)              : Contains logging dictionary and base logger
        debug_detection         (dict)              : Contains computed parameters pertaining to hvac
    """

    # Initialize the logger for this function

    logger_base = logger_pass.get("logger").getChild("compute_features_for_sh_detection")
    logger_hvac = logging.LoggerAdapter(logger_base, logger_pass.get("logging_dict"))

    # Prepare the logger pass dictionary

    logger_pass['logger'] = logger_base

    # Extract features for sh detection

    month_params = get_month_params(hvac_input, hvac_params['detection'], logger_pass)
    debug_tstat = get_tstat_values(hvac_input, hvac_params, logger_pass)
    user_params = get_user_params(month_params, hvac_params, vacation_monthly, logger_pass)

    # Populate computed features in the debug dictionary

    debug_detection['hdd']['debug_tstat'] = debug_tstat
    debug_detection['hdd']['month_params'] = month_params
    debug_detection['hdd']['SH_log_reg_features'] = user_params

    logger_hvac.debug('SH detection feature computation completed |')

    return None


def get_month_params(input_data, hvac_params_detection, logger_pass):

    """
    Function to compute monthly features needed for SH logistic regression detection model

    Parameters:
        input_data              (np.ndarray)        : Feature matrix for hvac_module
        hvac_params_detection   (np.ndarray)        : The predicted vacation epochs
        logger_pass             (dict)              : Contains logging dictionary and base logger

    Return:
        df_stats                (pd.DataFrame)      : The aggregated vacation stats
    """

    # Initialize the logger for this function

    logger_base = logger_pass.get("logger").getChild("get_month_params")
    logger_hvac = logging.LoggerAdapter(logger_base, logger_pass.get("logging_dict"))

    # Prepare the logger pass dictionary

    logger_pass['logger'] = logger_base

    # Initialize a data frame

    col_to_extract = [Cgbdisagg.INPUT_BILL_CYCLE_IDX, Cgbdisagg.INPUT_DAY_IDX, Cgbdisagg.INPUT_HOD_IDX,
                      Cgbdisagg.INPUT_CONSUMPTION_IDX, Cgbdisagg.INPUT_TEMPERATURE_IDX]

    df_col_headings = ['bc_epoch', 'day_epoch', 'hour_of_day', 'net', 'temperature']

    df = pd.DataFrame(input_data[:, col_to_extract], columns=df_col_headings)

    # Extract set point for AC and SH

    set_point_for_cdd = hvac_params_detection['AC']['SETPOINT_FOR_DECISION_TREE']
    set_point_for_hdd = hvac_params_detection['SH']['SETPOINT_FOR_DECISION_TREE']

    logger_hvac.info('computing monthly CDD/HDD based on config | '
                     'set_point_for_cdd: {}, set_point_for_hdd: {}'.format(set_point_for_cdd, set_point_for_hdd))

    # Add cdd hdd to the data frame

    df['cdd'] = [max(x - set_point_for_cdd, 0) for x in df['temperature']]
    df['hdd'] = [max(set_point_for_hdd - x, 0) for x in df['temperature']]

    # Compute sum and count of days across the complete data frame

    monthly = df.groupby('bc_epoch').agg({'cdd': 'sum',
                                          'hdd': 'sum',
                                          'net': 'sum',
                                          'day_epoch': 'nunique'}).rename({'cdd': 'cdd_sum',
                                                                           'hdd': 'hdd_sum',
                                                                           'net': 'net_sum',
                                                                           'day_epoch': 'num_days'}, axis=1)

    # Initialize dictionary containing month params

    month_params = {
        'month_cdd': monthly['cdd_sum'],
        'month_hdd': monthly['hdd_sum'],
        'month_net': monthly['net_sum'],
        'month_num_days': monthly['num_days']
    }

    return month_params


def get_tstat_values(hvac_input, hvac_params, logger_pass):

    """
    Function to compute t-statistics at hourly level for extreme temperature energy vs mid-temperature energy
    This information is used to compute features for the SH user level detection model

    Parameters:
        hvac_input              (np.ndarray)          : Feature matrix for hvac_module
        hvac_params             (dict)                : Contains config parameters for hvac
        logger_pass             (dict)                : Contains logging dictionary and base logger

    Return:
        debug_tstat             (dict)                : T-statistics
    """

    # Initialize the logger for this function

    logger_base = logger_pass.get("logger").getChild("get_tstat_values")
    logger_hvac = logging.LoggerAdapter(logger_base, logger_pass.get("logging_dict"))

    # Prepare the logger pass dictionary

    logger_pass['logger'] = logger_base

    # Initialize data frame with relevant columns from input data

    col_to_extract = [Cgbdisagg.INPUT_BILL_CYCLE_IDX, Cgbdisagg.INPUT_DAY_IDX, Cgbdisagg.INPUT_HOD_IDX,
                      Cgbdisagg.INPUT_EPOCH_IDX, Cgbdisagg.INPUT_TEMPERATURE_IDX, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    bc_col_headings = ['bc_epoch', 'day_epoch', 'hour_of_day', 'epoch', 'temperature', 'net']

    df = pd.DataFrame(hvac_input[:, col_to_extract], columns=bc_col_headings)

    # Initialize different variables to compute values as needed

    sampling_rate = df['epoch'].diff().mode()

    df['day_max_temp'] = df[['day_epoch', 'temperature']].groupby('day_epoch')['temperature'].transform('max')
    df['day_min_temp'] = df[['day_epoch', 'temperature']].groupby('day_epoch')['temperature'].transform('min')

    max_temps = df[['day_epoch', 'day_max_temp']].drop_duplicates()
    min_temps = df[['day_epoch', 'day_min_temp']].drop_duplicates()

    # HIGHLY UNOPTIMIZED CODE HERE

    min_temp_bound = None

    # Getting peak temperature for hdd days

    for min_temp_bound in range(hvac_params.get('detection').get('SH').get('MIN_TEMP_BOUND_LOWER'),
                                hvac_params.get('detection').get('SH').get('MIN_TEMP_BOUND_UPPER'),
                                hvac_params.get('detection').get('SH').get('MIN_TEMP_BOUND_STEP')):

        if (np.sum(min_temps['day_min_temp'] < min_temp_bound) >
                hvac_params.get('detection').get('SH').get('MIN_DATA_POINTS_THRESH')):
            break

    # Getting peak temperature for cdd days

    for max_temp_bound in range(hvac_params.get('detection').get('SH').get('MAX_TEMP_BOUND_UPPER'),
                                hvac_params.get('detection').get('SH').get('MAX_TEMP_BOUND_LOWER'),
                                hvac_params.get('detection').get('SH').get('MAX_TEMP_BOUND_STEP')):

        if (np.sum(max_temps['day_max_temp'] > max_temp_bound) >
                hvac_params.get('detection').get('SH').get('MIN_DATA_POINTS_THRESH')):
            break

    # Array to store t-statistics

    mid_vals = None
    extreme_vals = None

    tstats = []

    for hour_of_day in range(0, Cgbdisagg.HRS_IN_DAY, 1):

        # filtering days and returning values for corresponding hour

        mid_vals, extreme_vals = \
            return_filtered_data_daily(df, hvac_params, hvac_params.get('detection').get('SH').get('LOWER_TEMP_FILTER'),
                                       hvac_params.get('detection').get('SH').get('UPPER_TEMP_FILTER'), min_temp_bound,
                                       hour_of_day)

        mid_vals = mid_vals[~mid_vals.isna()]
        extreme_vals = extreme_vals[~extreme_vals.isna()]

        # getting tstat for each hour

        tstat = ttest_unequal_variances(extreme_vals, mid_vals,
                                        hvac_params.get('detection').get('SH').get('TTEST_THRESH'), sampling_rate)

        if np.isscalar(tstat):
            tstats.append(tstat)
        else:
            tstats.append(tstat.values[0])

        if len(mid_vals) == 0:
            mid_vals = -1

        if len(extreme_vals) == 0:
            extreme_vals = -1

    if any(np.isnan(tstats)):
        logger_hvac.debug('There is at least one nan in tstats vector, get_tstat_values function |')

    debug_tstat = dict({})
    debug_tstat['mid_vals'] = mid_vals
    debug_tstat['extreme_vals'] = extreme_vals
    debug_tstat['t_stats'] = tstats
    debug_tstat['t_stat_ma_max'], debug_tstat['t_stat_ma_avg'] = max_moving_mean(tstats, window_size=4)

    return debug_tstat


def return_filtered_data_daily(df, hvac_config, min_mid_temp, max_mid_temp, extreme_temp, hour_of_day):

    """
    Filters data according to mid temperatures and extreme temperatures by hour

    Parameters:
        df                  (pd.DataFrame)      : daily aggregates of temperature
        hvac_config         (dict)              : Contains config parameters for hvac
        min_mid_temp        (int)               : minimum mid temperature threshold
        max_mid_temp        (int)               : maximum mid temperature threshold
        extreme_temp        (int)               : extreme temperature threshold
        hour_of_day         (int)               : hour in day

    Return:
        mid_vals            (pd.Series)         : energy at mid temperatures
        extreme_vals        (pd.Series)         : energy at extreme temperatures
    """

    recursive_temp_inc = int(hvac_config.get('detection').get('SH').get('FILTER_DATA_RECURSIVE_TEMP_INC'))
    mid_idx = (df['day_min_temp'] > min_mid_temp) & (df['day_max_temp'] <= max_mid_temp) & (df['hour_of_day'] ==
                                                                                            hour_of_day)

    # Indices for extremely cold temperatures

    extreme_idx = (df['day_min_temp'] < extreme_temp) & (df['hour_of_day'] == hour_of_day)

    mid_vals = df.loc[mid_idx, 'net']
    extreme_vals = df.loc[extreme_idx, 'net']

    if (len(mid_vals) < int(hvac_config.get('detection').get('SH').get('FILTER_DATA_MID_VAL_LOWER_THRESH'))
            & min_mid_temp > int(hvac_config.get('detection').get('SH').get('FILTER_DATA_MIN_MID_TEMP_UPPER_THRESH'))):

        mid_vals, extreme_vals = return_filtered_data_daily(df, hvac_config, min_mid_temp - recursive_temp_inc,
                                                            max_mid_temp + recursive_temp_inc, extreme_temp,
                                                            hour_of_day)

    return mid_vals, extreme_vals


def ttest_unequal_variances(extreme_vals, mid_vals, threshold, sampling_rate):

    """
    Filters data according to mid temperatures and extreme temperatures by hour

    Parameters:
        extreme_vals        (pd.Series)         : energy value at extreme temperature
        mid_vals            (pd.Series)         : energy value at mid temperature
        threshold           (int)               : threshold for t test
        sampling_rate       (int)               : The time period in seconds at which the data is recorded

    Return:
        t_stat              (float)             : t-statistic for extreme values vs mid values
    """

    # Remove nan values from the arrays

    extreme_vals = extreme_vals[~extreme_vals.isna()]
    mid_vals = mid_vals[~mid_vals.isna()]

    x_len = len(extreme_vals)
    y_len = len(mid_vals)

    # Compute and return tstat value

    if x_len == 0 or y_len == 0:
        tstat = np.NaN
        return tstat
    else:
        x_mean = np.mean(extreme_vals)
        x_std = np.std(extreme_vals)

        y_mean = np.mean(mid_vals)
        y_std = np.std(mid_vals)

        pool_std = np.sqrt((x_std ** 2 / x_len) + (y_std ** 2 / y_len))

        if pool_std == 0 or np.isnan(pool_std):
            tstat = np.NaN
            return tstat

        tstat = (x_mean - y_mean - threshold * sampling_rate / Cgbdisagg.SEC_IN_HOUR) / pool_std

    return tstat


def max_moving_mean(x, window_size=4):

    """
    Filters data according to mid temperatures and extreme temperatures by hour

    Parameters:
        x                  (list)               : list of t-statistics
        window_size        (int)                : size of rolling window

    Return:
        max_mov_mean       (float)              : maximum of moving average
        means              (pd.Series)          : vector of means
    """

    # Appending values at end and start might change depending on window.

    means = pd.Series(x).rolling(window=window_size, center=True).mean()
    max_mov_mean = np.max(means)

    # If the mean is nan we initialize it to zero

    if np.isnan(max_mov_mean):
        max_mov_mean = 0

    return max_mov_mean, means


def get_user_params(month_params, hvac_config, vacation_monthly, logger_pass):

    """
    Function to compute features for SH logistic regression model at user level

    Parameters:
        month_params            (pd.DataFrame)      : monthly
        hvac_config             (dict)              : Contains config parameters for hvac
        vacation_monthly        (np.ndarray)        : aggregated vacation at monthly level
        logger_pass             (dict)              : Contains logging dictionary and base logger

    Return:
        user_params             (dict)              : features for SH detection model at user level
    """

    # Initialize the logger for this function

    logger_base = logger_pass.get("logger").getChild("get_user_params")
    logger_hvac = logging.LoggerAdapter(logger_base, logger_pass.get("logging_dict"))

    # Prepare the logger pass dictionary

    logger_pass['logger'] = logger_base

    user_params = dict({})
    user_params['SH'] = {}

    # Convert all pandas values to numpy

    month_hdd = month_params['month_hdd'].values
    month_cdd = month_params['month_cdd'].values
    month_net = month_params['month_net'].values
    month_num_days = month_params['month_num_days'].values
    percent_vacation = vacation_monthly['percent_vacation'].values

    month_valid_num_days = np.floor(month_num_days - month_num_days * percent_vacation)

    month_net = (month_valid_num_days / 30) * month_net

    good_months = month_valid_num_days > int(hvac_config.get('detection').get('SH').get('MONTH_VALID_NUM_DAYS_THRESH'))

    # default parameters, will

    user_params['SH']['corr'] = 0
    user_params['SH']['max_minus_min_hddcdd'] = 0
    user_params['SH']['corr_month_selection'] = 0
    user_params['SH']['max_minus_min_ms'] = 0

    if sum(good_months) > 1:

        logger_hvac.info('more than 1 good month (>25 non vacation days) data found,'
                         'entering SH detection model feature computation |')

        # compute space heating features for SH user level detection model

        imp_months = (month_hdd > month_cdd) & good_months

        if sum(imp_months) == 0:

            imp_months = (month_hdd == max(month_hdd)) & good_months
            logger_hvac.debug('0 months found that were good (>25 non vacation days) and month_hdd > month_cdd,'
                              ' can not compute monthly SH features |')

        if sum(imp_months) > 0:
            logger_hvac.info('computing monthly SH detection features |')

            # correlation between consumption and cold weather

            corr_array = np.hstack([month_hdd[imp_months].reshape(-1, 1), month_net[imp_months].reshape(-1, 1)])

            if corr_array.shape[0] == 1:
                user_params['SH']['corr'] = np.corrcoef(corr_array, rowvar=False)
            else:
                user_params['SH']['corr'] = np.corrcoef(corr_array, rowvar=False)[0, 1]

            # consumption difference between largest and smallest consumption months

            user_params['SH']['max_minus_min_hddcdd'] = max(month_net[imp_months]) - min(month_net[imp_months])

            # consecutive important months

            consec_imp_months = (imp_months | pd.Series(imp_months).shift(1).fillna(False) |
                                 pd.Series(imp_months).shift(-1).fillna(False))

            imp_months_ms = consec_imp_months & good_months

            corr_array = np.hstack([month_hdd[imp_months_ms].reshape(-1, 1), month_net[imp_months_ms].reshape(-1, 1)])

            if corr_array.shape[0] == 1:
                user_params['SH']['corr_month_selection'] = np.corrcoef(corr_array, rowvar=False)
            else:
                user_params['SH']['corr_month_selection'] = np.corrcoef(corr_array, rowvar=False)[0, 1]

            user_params['SH']['max_minus_min_ms'] = max(month_net[imp_months_ms]) - min(month_net[imp_months_ms])

            if max(month_net[imp_months]) < max(month_net[imp_months_ms]):
                user_params['SH']['max_minus_min_ms'] = user_params['SH']['max_minus_min_hddcdd']
    else:
        logger_hvac.info('can not compute monthly features for SH detection model,'
                         ' less than 1 good month (>25 non vacation days) of data |')

    return user_params
