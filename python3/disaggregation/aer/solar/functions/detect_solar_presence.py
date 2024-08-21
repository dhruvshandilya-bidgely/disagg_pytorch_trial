"""
Author - Akshay Gupta
Date - 3rd Oct 2022
This file has code for run solar disaggregation module for a user

"""

# Import python packages
import logging
import numpy as np
from copy import deepcopy
import warnings
from sklearn.metrics import r2_score
from numpy import dot
from numpy.linalg import norm
from sklearn import preprocessing
import pandas as pd
import math
import datetime
from python3.utils.time.get_time_diff import get_time_diff

# Import functions from within the project
from python3.disaggregation.aer.solar.functions.curve_estimation import get_irradiance
from python3.config.Cgbdisagg import Cgbdisagg
warnings.filterwarnings("ignore")


def get_detection_metrics(solar_detection_config, input_array, solar_presence, logger_base):
    """
    Function to return metrics such as irradiance, start/end time and state of solar panel presence

    Parameters:
        solar_detection_config  (dict)                  : Dictionary containing solar configurations
        input_array             (np.ndarray)            : Input thirteen column matrix
        logger_base             (object)                : Logger object
        solar_presence          (int)                   : Indicates solar presence

    Returns:
        irradiance              (numpy.ndarray)         : irradiance array
        start_date              (int)                   : Start date of solar panel presence
        end_date                (int)                   : End date of solar panel presence
        solar_panel_status                    (str)                   : Solar panel present throughout or installation/removal
    """

    # Taking new logger base for this module
    logger_local = logger_base.get("logger").getChild("get_solar_presence")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_base.get("logging_dict")}

    if not solar_presence:
        start_date, end_date = int(input_array[0, Cgbdisagg.INPUT_EPOCH_IDX]), int(input_array[-1, Cgbdisagg.INPUT_EPOCH_IDX])
        solar_panel_status = None
        irradiance = np.zeros_like(input_array[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        return irradiance, start_date, end_date, solar_panel_status

    # Create copy of input array
    input_data = deepcopy(input_array)

    # Create irradiance array
    irradiance = np.zeros_like(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # Create sun presence array
    sun_presence = np.logical_and(
        input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] >= input_data[:, Cgbdisagg.INPUT_SUNRISE_IDX],
        input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] <= input_data[:, Cgbdisagg.INPUT_SUNSET_IDX])

    # Create epoch array only during sun times
    epoch_time = input_data[sun_presence, Cgbdisagg.INPUT_EPOCH_IDX]

    # Get location information
    latitude = solar_detection_config.get('latitude')
    longitude = solar_detection_config.get('longitude')
    user_timezone = solar_detection_config.get('timezone')

    # Calculate Irradiance for only sun presence hours
    irradiance[sun_presence] = get_irradiance(epoch_time, latitude, longitude, user_timezone, logger_pass)
    start_date, end_date, solar_panel_status = get_solar_timings(np.c_[input_data, sun_presence], Cgbdisagg.INPUT_DIMENSION,
                                                                 solar_detection_config, logger_pass)

    return irradiance, start_date, end_date, solar_panel_status


def get_solar_timings(input_array, sun_presence_col, solar_co, logger_base):

    """
     Get start/end timings for solar generation

     Parameters:
        input_array             (pandas.dataframe)      : Pandas DataFrame containing 28 column matrix
        sun_presence_col        (int)                   : Sun presence column
        solar_co                (dict)                  : Dict with all config info required by solar module
        logger_base             (object)                : Logger object

     Returns:
        start_date              (int)                    : Start date of solar panel presence
        end_date                (int)                    : End date of solar panel presence
        solar_panel_status      (str)                    : Solar panel present throughout or installation/removal
    """
    # Set up logger
    logger_local = logger_base.get("logger").getChild("get_solar_timings")
    logger = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # Create copy of input data
    input_df = pd.DataFrame(deepcopy(input_array))

    # Create mask for sun presence and negative values
    sun_presence_mask = input_df[sun_presence_col] == 1
    neg_consumption_mask = input_df.loc[sun_presence_mask, Cgbdisagg.INPUT_CONSUMPTION_IDX] < 0
    input_df[sun_presence_col+1] = neg_consumption_mask*1

    # Calculate weighted mean for negative values in each week
    neg_ew_mean = input_df.pivot_table(index=Cgbdisagg.INPUT_WEEK_IDX, values=sun_presence_col+1, aggfunc=np.sum).ewm(alpha=1 / 3).mean()
    neg_ew_mean = (neg_ew_mean > 10).cumsum()

    # Month in seconds
    sec_in_months = Cgbdisagg.DAYS_IN_MONTH*Cgbdisagg.SEC_IN_DAY

    logger.debug('Calculating start/end date for solar detection |')

    # Start and End Dates
    start = neg_ew_mean[neg_ew_mean>0].drop_duplicates(keep='last').idxmin().values[0]
    start_date = max(input_df.loc[:, Cgbdisagg.INPUT_EPOCH_IDX].min(), start - sec_in_months)
    end = neg_ew_mean[neg_ew_mean>0].drop_duplicates(keep='first').idxmax().values[0]
    end_date = min(input_df.loc[:, Cgbdisagg.INPUT_EPOCH_IDX].max(), end + sec_in_months)
    if end_date<start_date:
        start_date, end_date = input_df[Cgbdisagg.INPUT_EPOCH_IDX].iloc[0], input_df[Cgbdisagg.INPUT_EPOCH_IDX].iloc[-1]

    # Specify if solar panel was present throughout or installed/removed
    if start_date > input_df.loc[:, Cgbdisagg.INPUT_EPOCH_IDX].min() + sec_in_months:
        solar_panel_status = solar_co.get('solar_panel_presence', {}).get('installation')

    elif end_date < input_df.loc[:, Cgbdisagg.INPUT_EPOCH_IDX].max() - sec_in_months:
        solar_panel_status = solar_co.get('solar_panel_presence', {}).get('removal')

    else:
        solar_panel_status = solar_co.get('solar_panel_presence', {}).get('present_throughout')

    logger.debug('Successfully calculated start/end date for solar detection |')

    return int(start_date), int(end_date), solar_panel_status


def get_negative_percentage(input_data, sun_presence_col):

    """
     Get percentage negative/zero values

     Parameters:
         input_data              (pandas.dataframe)      : Pandas DataFrame containing 28 column matrix
         sun_presence_col        (int)                   : Sun presence column

    Returns:
        neg_perc_day             (float)                 : Percentage of negative values during the day
        neg_perc_night           (float)                 : Percentage of negative values during the night
        zero_perc_day            (float)                 : Percentage of zero values during the day
        zero_perc_night          (float)                 : Percentage of zero values during the night
    """

    # Create copy of input data
    input_df = deepcopy(input_data)

    # Create mask for sun presence and negative values
    sun_presence_mask = input_df[sun_presence_col] == 1
    neg_consumption_mask = input_df[Cgbdisagg.INPUT_CONSUMPTION_IDX] < 0
    zero_consumption_mask = input_df[Cgbdisagg.INPUT_CONSUMPTION_IDX] == 0

    # Calculate negative percentage during day/night
    neg_perc_day = 100 * len(input_df[(sun_presence_mask) & (neg_consumption_mask)]) / len(input_df[sun_presence_mask])
    neg_perc_night = 100 * len(input_df[(~sun_presence_mask) & (neg_consumption_mask)]) / len(input_df[~sun_presence_mask])
    zero_perc_day = len(input_df[(sun_presence_mask) & (zero_consumption_mask)]) / len(input_df[sun_presence_mask])
    zero_perc_night = len(input_df[(~sun_presence_mask) & (zero_consumption_mask)]) / len(input_df[~sun_presence_mask])

    return zero_perc_day, zero_perc_night, neg_perc_day, neg_perc_night


def get_irradiance_metrics(input_data, sun_presence_col, solar_config, logger_base):
    """
     Get metrics for irradiance

     Parameters:
        input_data                                 (pandas.dataframe)      : Pandas DataFrame containing 28 column matrix
        sun_presence_col                           (int)                   : Sun presence column
        solar_config                               (dict)                  : Dict with all config info required by solar module
        logger_base                                (object)                : Logger object

     Returns:
        monthly_min_curve                          (pandas.dataframe)     : Pandas DataFrame containing monthly min values for each hour
        cos_sim_monthly_irr_similarity             (list of float)        : Monthly irradiance similarity values
        cos_sim_hourly_irr                         (list of float)        : Hourly irradiance similarity values
        irradiance                                 (numpy ndarray)        : Epoch-wise irradiance
    """

    # Taking new logger base for this module
    logger_local = logger_base.get("logger").getChild("get_solar_presence")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_base.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # Create copy of input data
    input_df = deepcopy(input_data)

    # Create mask for sun presence
    sun_presence_mask = input_df[sun_presence_col] == 1

    # Create irradiance array
    irradiance = np.zeros_like(input_df[Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # Create epoch array only during sun times
    epoch_time = np.array(input_df.loc[sun_presence_mask, Cgbdisagg.INPUT_EPOCH_IDX])

    # Get location information
    latitude = solar_config.get('latitude')
    longitude = solar_config.get('longitude')
    user_timezone = solar_config.get('timezone')

    # Calculate Irradiance for only sun presence hours and convert nans to 0
    irradiance[sun_presence_mask] = get_irradiance(epoch_time, latitude, longitude, user_timezone, logger_pass)
    irradiance[np.isnan(irradiance)] = 0
    irradiance_raw = deepcopy(irradiance)

    # Load min-max scaler
    min_max_scaler = preprocessing.MinMaxScaler()

    # Concat irradiance array to dataframe
    irradiance_col = Cgbdisagg.INPUT_DIMENSION + 1
    irradiance = pd.Series(min_max_scaler.fit_transform(np.array(irradiance).reshape(-1, 1)).T[0])
    input_df[irradiance_col] = irradiance

    # Create date-to-timestamp dict to efficiently map month
    month_col = Cgbdisagg.INPUT_DIMENSION + 2
    unique_days = input_df[Cgbdisagg.INPUT_DAY_IDX].unique()
    month_dict = {i: j for i, j in zip(unique_days, pd.to_datetime(unique_days, unit='s'))}
    input_df[month_col] = input_df[Cgbdisagg.INPUT_DAY_IDX].map(month_dict).dt.month

    # Create pivot table to get mean values of irradiance monthly for each hour of the day
    monthly_irradiance = input_df.dropna(subset=[Cgbdisagg.INPUT_CONSUMPTION_IDX]).pivot_table(index=[month_col],
                                                                                               values=irradiance_col, columns=Cgbdisagg.INPUT_HOD_IDX,
                                                                                               aggfunc=np.mean, fill_value=0)
    monthly_irradiance = pd.DataFrame(min_max_scaler.fit_transform(monthly_irradiance.T)).T
    monthly_irradiance = (1 - monthly_irradiance.T)

    # Create pivot table to get min values of consumption monthly for each hour of the day
    monthly_min_curve = input_df.dropna(subset=[Cgbdisagg.INPUT_CONSUMPTION_IDX]).pivot_table(index=[month_col], values=Cgbdisagg.INPUT_CONSUMPTION_IDX,
                                                                                              columns=Cgbdisagg.INPUT_HOD_IDX,
                                                                                              aggfunc=np.min, fill_value=0)
    monthly_min_curve = pd.DataFrame(min_max_scaler.fit_transform(monthly_min_curve.T))

    # Calculate similarity between Monthly Irradiance and Minimum curve
    cos_sim_monthly_irr_similarity = []
    for i in range(len(monthly_irradiance)):
        a = monthly_irradiance.iloc[i].values
        b = monthly_min_curve.iloc[i].values
        cos_sim_monthly_irr_similarity.append(dot(a, b) / (norm(a) * norm(b)))

    # Calculate similarity between Hourly Irradiance and Minimum curve
    cos_sim_hourly_irr = []
    for i in range(len(monthly_irradiance.columns)):
        a = monthly_irradiance.iloc[:, i].values
        b = monthly_min_curve.iloc[:, i].values
        cos_sim_hourly_irr.append(dot(a, b) / (norm(a) * norm(b)))

    logger.info('Irradiance metrics calculated successfully |  ')

    return irradiance_raw, monthly_min_curve, cos_sim_monthly_irr_similarity, cos_sim_hourly_irr


def get_hourwise_difference_metrics(input_data, sun_presence_col, monthly_min_curve_df, uuid_regression_stats):
    """
     Get metrics for hourwise differences in consumption

     Parameters:
        input_data                                 (pandas.dataframe)      : Pandas DataFrame containing 28 column matrix
        sun_presence_col                           (int)                   : Sun presence column
        monthly_min_curve_df                       (pandas.dataframe)      : Pandas DataFrame containing monthly min values for each hour
        uuid_regression_stats                      (dictionary of float)   : Statistics for regression output

     Returns:
        uuid_regression_stats                      (dictionary of float)   : Statistics for regression output
    """

    # Create copy of monthly_min_curve
    monthly_min_curve = deepcopy(monthly_min_curve_df)

    # Get monthly hourwise difference
    monthly_min_curve_diff = monthly_min_curve - monthly_min_curve.shift()

    # Calculate Hour-wise Metrics
    uuid_regression_stats['max_diff_mean'] = monthly_min_curve_diff.T.max().mean()
    uuid_regression_stats['max_diff_var'] = monthly_min_curve_diff.T.max().var()
    uuid_regression_stats['min_diff_mean'] = monthly_min_curve_diff.T.min().mean()
    uuid_regression_stats['min_diff_var'] = monthly_min_curve_diff.T.min().var()
    uuid_regression_stats['max_min_diff_mean'] = abs(
        monthly_min_curve_diff.T.max() + monthly_min_curve_diff.T.min()).mean()
    uuid_regression_stats['max_min_diff_var'] = abs(
        monthly_min_curve_diff.T.max() + monthly_min_curve_diff.T.min()).var()

    return uuid_regression_stats


def get_percentile_regression_stats(input_data, sun_presence_col, uuid_regression_stats):
    """
     Get statistics on regression based on minimum curves

     Parameters:
        input_data                                 (pandas.dataframe)      : Pandas DataFrame containing 28 column matrix
        sun_presence_col                           (int)                   : Sun presence column
        uuid_regression_stats                      (dictionary of float)   : Statistics for regression output

     Returns:
        uuid_regression_stats                      (dictionary of float)   : Statistics for regression output
    """
    # Create copy of input data
    input_df = deepcopy(input_data)

    # Create mask for sun presence
    sun_presence_mask = input_df[sun_presence_col] == 1

    # Get max/min hour for regression
    min_hour = input_df[sun_presence_mask][Cgbdisagg.INPUT_HOD_IDX].min() - 1
    max_hour = input_df[sun_presence_mask][Cgbdisagg.INPUT_HOD_IDX].max() + 1

    # Downsample to get minimum values for each hour, each day
    pivot_index = [Cgbdisagg.INPUT_DAY_IDX, Cgbdisagg.INPUT_HOD_IDX]
    downsampled_input_data = input_df.pivot_table(index=pivot_index, values=Cgbdisagg.INPUT_CONSUMPTION_IDX,
                                                  aggfunc=np.min).reset_index()

    # Get regression stats for 0-0.03 percentiles
    for percentile in np.arange(0, 0.04, 0.01):

        pivot_columns = [Cgbdisagg.INPUT_HOD_IDX, Cgbdisagg.INPUT_CONSUMPTION_IDX]
        temp = downsampled_input_data[pivot_columns].pivot_table(index=Cgbdisagg.INPUT_HOD_IDX,
                                                                 aggfunc=lambda x: x.quantile(percentile)).reset_index()

        # Sun presence for downsampled data
        normalised = max_hour - min_hour
        sun_presence_downsampled = (temp[Cgbdisagg.INPUT_HOD_IDX] - min_hour) / normalised
        distance_from_sunrise = np.sin(math.pi * sun_presence_downsampled)
        sun_presence_downsampled = (sun_presence_downsampled > 0) & (sun_presence_downsampled < 1)

        # Create mask
        sun_presence_mask_downsampled = sun_presence_downsampled == 1

        # X, Y for regression
        x_reg = distance_from_sunrise[sun_presence_mask_downsampled]
        y_reg = -temp[sun_presence_mask_downsampled][Cgbdisagg.INPUT_CONSUMPTION_IDX]

        reg_df = pd.DataFrame(np.c_[x_reg, y_reg], columns=['x', 'y'])

        z3 = np.polyfit(reg_df['x'], reg_df['y'], 1)
        r_square3 = r2_score(reg_df['y'], z3[0] * reg_df['x'] + z3[1])
        slope3, intercept3 = z3[0], z3[1]
        r_square3 = np.round(r_square3, 2)

        uuid_regression_stats['slope_' + str(percentile)] = slope3
        uuid_regression_stats['intercept_' + str(percentile)] = intercept3
        uuid_regression_stats['r_square_' + str(percentile)] = r_square3

    return uuid_regression_stats


def get_solar_presence(input_array, disagg_input_object, confidence_cnn, solar_config, logger_base):

    """
    Detect solar presence

    Parameters:
        input_array             (numpy.ndarray)         : Numpy array containing 28 column matrix
        disagg_input_object     (dict)                  : disagg input object
        confidence_cnn              (float)             : CNN solar detection output
        solar_config            (dict)                  : Dict with all config info required by solar module
        logger_base             (object)                : Logger object

    Returns:
        irradiance              (numpy.ndarray)         : irradiance array
        solar_presence          (int)                   : Solar presence flag
        confidence              (float)                 : Confidence of lgb model
        start_date              (int)                   : Start date of solar panel presence
        end_date                (int)                   : End date of solar panel presence
        solar_panel_status      (str)                   : Solar panel present throughout or installation/removal
    """

    # Taking new logger base for this module
    logger_local = logger_base.get("logger").getChild("get_solar_presence")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_base.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    logger.debug('Detecting solar presence for the user |')

    time = datetime.datetime.now()

    #Make copy of the original data
    input_data = pd.DataFrame(deepcopy(input_array))

    #Capture metrics for the user
    uuid_regression_stats = {}

    start_time = datetime.datetime.now()
    logger.debug('Calculating parameters for detecting solar presence|')

    #Neg/Zero Percentage
    zero_perc_day, zero_perc_night, neg_perc_day, neg_perc_night = get_negative_percentage(input_data, Cgbdisagg.INPUT_DIMENSION)

    uuid_regression_stats['neg_perc_day'] = neg_perc_day
    uuid_regression_stats['neg_perc_night'] = neg_perc_night
    uuid_regression_stats['zero_perc_day'] = zero_perc_day
    uuid_regression_stats['zero_perc_night'] = zero_perc_night

    # Irradiance Metrics
    irradiance, monthly_min_curve, cos_sim_monthly_irr_similarity, cos_sim_hourly_irr = \
        get_irradiance_metrics(input_data, Cgbdisagg.INPUT_DIMENSION, solar_config, logger_pass)

    # Calculate hourwise difference in minimum valuex
    uuid_regression_stats = get_hourwise_difference_metrics(input_data, Cgbdisagg.INPUT_DIMENSION, monthly_min_curve, uuid_regression_stats)

    # Get percentile regression statistics
    uuid_regression_stats = get_percentile_regression_stats(input_data, Cgbdisagg.INPUT_DIMENSION, uuid_regression_stats)

    # Add rest of the regression stats
    uuid_regression_stats['confidence'] = confidence_cnn
    uuid_regression_stats['cos_sim_monthly_irr_similarity_mean'] = np.nanmean(cos_sim_monthly_irr_similarity)
    uuid_regression_stats['cos_sim_monthly_irr_similarity_var'] = np.nanvar(cos_sim_monthly_irr_similarity)
    uuid_regression_stats['cos_sim_hourly_irr_mean'] = np.nanmean(cos_sim_hourly_irr)
    uuid_regression_stats['cos_sim_hourly_irr_var'] = np.nanvar(cos_sim_hourly_irr)

    logger.info('Successfully calculated parameters for detecting solar presence |')

    end_time = datetime.datetime.now()
    parameter_calculation_time = get_time_diff(start_time, end_time)
    logger.info('Timing: Successfully calculated parameters for solar presence | %0.3f', parameter_calculation_time)
    logger.info('Solar presence parameters | {}'.format(uuid_regression_stats))

    zero_percentage_threshold = 0 if uuid_regression_stats['neg_perc_day'] >= 1 else uuid_regression_stats['zero_perc_day'] - uuid_regression_stats['zero_perc_night']
    zero_pass = zero_percentage_threshold >= solar_config.get('solar_disagg').get('zero_suntime_thresh')

    if zero_perc_day + uuid_regression_stats['neg_perc_day'] <= 0:
        confidence = 0
        sun_presence = 0
        start_date = np.nanmin(input_array[:, Cgbdisagg.INPUT_EPOCH_IDX])
        end_date = np.nanmax(input_array[:, Cgbdisagg.INPUT_EPOCH_IDX])
        solar_panel_status = None
        return np.array(irradiance), sun_presence, confidence, start_date, end_date, solar_panel_status

    del uuid_regression_stats['zero_perc_day']
    del uuid_regression_stats['zero_perc_night']

    # Import LightGBM Model
    lgb = disagg_input_object.get('loaded_files', {}).get('solar_files', {}).get('detection_lgb_model', {})

    if lgb and not zero_pass:
        eval_array = np.array(list(uuid_regression_stats.values())).reshape(1,-1)
        confidence = round(lgb.predict_proba(eval_array)[0][1], 2)
        confidence_threshold = solar_config.get('solar_disagg',{}).get('lgbm_threshold',{})
        sun_presence = 1 if confidence>=confidence_threshold else 0
    else:
        confidence = np.round(confidence_cnn, 2)
        sun_presence = 1 if confidence >=.5 else 0

    # Get Solar Installation/Removal Date
    if sun_presence:
        start_date, end_date, solar_panel_status = get_solar_timings(input_data, Cgbdisagg.INPUT_DIMENSION, solar_config, logger_pass)

    else:
        start_date, end_date = input_array[0, Cgbdisagg.INPUT_EPOCH_IDX], input_array[-1, Cgbdisagg.INPUT_EPOCH_IDX]
        solar_panel_status = None

    end_time = datetime.datetime.now()
    solar_presence_time = get_time_diff(time, end_time)
    logger.info('Timing: Successfully ran solar presence detection | %0.3f', solar_presence_time)

    return np.array(irradiance), sun_presence, confidence, start_date, end_date, solar_panel_status
