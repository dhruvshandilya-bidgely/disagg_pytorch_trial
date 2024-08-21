"""
Author - Prasoon Patidar
Date - 0rd June 2020
Preprocess global input to create input for lifestyle modules
"""

# import python packages

import copy
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.logs_utils import log_prefix
from python3.utils.time.get_time_diff import get_time_diff
from python3.master_pipeline.preprocessing.downsample_data import downsample_data
from python3.analytics.lifestyle.functions.lifestyle_utils import get_day_level_2d_matrix
from python3.analytics.lifestyle.functions.lifestyle_utils import get_weekend_warrior_input


def prepare_lifestyle_input_data(lifestyle_input_object, disagg_input_object, disagg_output_object, logger_pass):

    """
    Parameters:
        lifestyle_input_object(dict)            : Dictionary containing lifestyle specific input
        disagg_input_object (dict)              : Dictionary containing all disagg inputs
        disagg_output_object(dict)              : Dictionary containing all disagg outputs
        logger_pass(dict)                       : Contains base logger and logging dictionary

    Returns:
        input_data(np.ndarray)                  : 2-D Array for all processed lifestyle inputs
        new_input_data_dict(dict)                : Dictionary containing index and other information for lifestyle input data
    """

    t_prepare_lifestyle_input_start = datetime.now()

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('prepare_lifestyle_input_data')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.info("%s Start preparing lifestyle input data", log_prefix('Generic'))

    # Retrieve config for raw_data and weather config

    raw_data_config = lifestyle_input_object.get('raw_data_config')
    weather_config = lifestyle_input_object.get('weather_config')

    # Copy raw input data from disagg input object

    original_input_data = copy.deepcopy(disagg_input_object.get('input_data'))

    nan_idx = np.isnan(original_input_data)
    original_input_data[nan_idx] = 0

    # This is copy of original input data from disagg input object, where nans are filled with zeros
    lifestyle_input_object['original_input_data'] = original_input_data

    input_data = copy.deepcopy(original_input_data)

    # Trim consumption values to remove outlier consumption

    max_percentile_val = raw_data_config.get('max_percentile_val')

    max_allowed_consumption_val = np.percentile(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], max_percentile_val)

    # TODO(Nisha): this is a wrong method, need to make corrections

    consumption_fill_value = np.percentile(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], max_percentile_val + 1)

    logger.debug("%s supress outlier value greater than %s Wh to %s Wh.",
                 log_prefix('Generic'), str(round(max_allowed_consumption_val,2)), str(round(consumption_fill_value,2)))

    input_data[
        input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > max_allowed_consumption_val, Cgbdisagg.INPUT_CONSUMPTION_IDX] = \
        consumption_fill_value

    # Remove poolpump consumption from input data

    special_outputs_poolpump = disagg_output_object.get('special_outputs').get('pp_consumption', None)

    pp_consumption_col = 1

    if special_outputs_poolpump is not None:
        # If poolpump is special outputs module, subtract poolpump from epoch output

        pp_epoch_estimate = special_outputs_poolpump[:, pp_consumption_col]

        pp_epoch_estimate[np.isnan(pp_epoch_estimate)] = 0.

        input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] -= pp_epoch_estimate

        logger.debug("%s poolpump consumption found, removed pp consumption", log_prefix('Generic'))

    # make sure there is no negative consumption due to pp removal

    input_data[input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] < 0, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0.

    # Trim number of days to max allowed days for lifestyle module

    max_day_val = np.max(input_data[:, Cgbdisagg.INPUT_DAY_IDX])
    min_day_val = np.min(input_data[:, Cgbdisagg.INPUT_DAY_IDX])


    max_allowed_epoch_difference = Cgbdisagg.SEC_IN_DAY * raw_data_config.get('num_days_limit')

    if (max_day_val - min_day_val) >= max_allowed_epoch_difference:
        min_allowed_day_val = max_day_val - max_allowed_epoch_difference
        input_data = input_data[input_data[:, Cgbdisagg.INPUT_DAY_IDX] >= min_allowed_day_val]
        min_day_val = min_allowed_day_val
        logger.debug("%s Trim input data betweek Min Day Epoch: %d and Max Day Epoch: %d",
                     log_prefix('Generic'), min_allowed_day_val, max_day_val)

    lifestyle_input_object.update({
        "max_day_timestamp" : max_day_val,
        "min_day_timestamp" : min_day_val})

    # get weekend warrior input data from original input data

    weekend_warrior_input_dict = get_weekend_warrior_input(disagg_input_object, lifestyle_input_object, logger_pass)

    # Downsample input data at hourly level

    input_data = downsample_data(input_data, Cgbdisagg.SEC_IN_HOUR)

    logger.debug("%s Shape of downsampled input data | %s", log_prefix('Generic'), str(input_data.shape))

    # Create 2d day*hour matrix for daily data input

    day_indices, day_input_data = get_day_level_2d_matrix(input_data, logger_pass)

    logger.debug("%s Shape of day-hour data | %s", log_prefix('Generic'), str(day_input_data.shape))

    # Get Seasonal info into input Data

    input_season_vals, monthly_season_vals, monthly_season_idx_dict = get_seasonal_info(lifestyle_input_object,
                                                                                        input_data, weather_config,
                                                                                        logger_pass)

    # get seasons for corresponding day indices

    day_datetime_vals = pd.to_datetime(day_indices, unit='s')

    day_month_vals = np.array((1e2 * day_datetime_vals.year + 1e0 * day_datetime_vals.month)).astype(int)

    season_vals_month_idx = monthly_season_idx_dict.get('MONTH_IDX')
    season_vals_season_idx = monthly_season_idx_dict.get('SEASON_IDX')

    day_season_vals =\
        np.array([monthly_season_vals[monthly_season_vals[:, season_vals_month_idx] ==
                                      month_val, season_vals_season_idx][0] for month_val in day_month_vals])

    # Init Seasonal Index, and write seasonal info in input_data

    season_idx = input_data.shape[1]

    # Init Weekday, Weekend Index

    logger.debug("%s Get Weekday/Weekend indicator", log_prefix('Generic'))

    weekday_idx = season_idx + 1
    saturday_day_id = raw_data_config.get('SATURDAY_DAY_ID')
    sunday_day_id = raw_data_config.get('SUNDAY_DAY_ID')

    # find corresponding weekday/weekend value for all rows

    is_weekday = np.ones(input_data.shape[0])
    is_weekday[(input_data[:, Cgbdisagg.INPUT_DOW_IDX] == saturday_day_id) |
               (input_data[:, Cgbdisagg.INPUT_DOW_IDX] == sunday_day_id)] = 0

    # Get new indices and extend input data
    new_idx_count = raw_data_config.get("NEW_IDX_COUNT")

    new_input_data_dict = {
        'SEASON_IDX'                : season_idx,
        'WEEKDAY_IDX'               : weekday_idx,
        'day_input_data_index'      : day_indices,
        'monthly_seasons'           : monthly_season_vals,
        'day_input_data_seasons'    : day_season_vals,
        'weekend_warrior_input_dict': weekend_warrior_input_dict
    }

    extended_input_data = np.empty((input_data.shape[0], input_data.shape[1] + +new_idx_count))
    extended_input_data[:, : -1 * new_idx_count] = input_data

    # Add Seasonal and weekday info in extended input data

    extended_input_data[:, season_idx] = input_season_vals

    extended_input_data[:, weekday_idx] = is_weekday

    logger.debug("%s Added Seasons and Weekday/Weekday information. Final input data shape | %s",
                 log_prefix('Generic'), str(extended_input_data.shape))

    t_prepare_lifestyle_input_end = datetime.now()

    logger.info("%s Preparing input for lifestyle took | %.3f s", log_prefix('Generic'),
                get_time_diff(t_prepare_lifestyle_input_start, t_prepare_lifestyle_input_end))

    return extended_input_data, day_input_data, new_input_data_dict


def get_seasonal_info(lifestyle_input_object, input_data, weather_config, logger_pass):

    """
    Parameters:
        lifestyle_input_object(dict)            : Dictionary containing lifestyle specific input
        input_data(np.ndarray)                  : Input data
        weather_config (dict)                   : Dictionary containing weather related config
        logger_pass(dict)                       : Contains base logger and logging dictionary

    Returns:
        season_vals(np.ndarray)               : 1-D Array for season value for all input_data rows
    """

    t_seasonal_info_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_seasonal_info')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Get seasonal info for month", log_prefix('Generic'))

    raw_data_config = lifestyle_input_object.get('raw_data_config')

    # get Enum for seasons from weather config

    seasons = weather_config.get('season')

    # Get HDD and CDD Values based on temperature setpoint in weather config

    temp_setpoint = weather_config.get('temperature_setpoint', None)

    input_temp_vals = input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]

    if temp_setpoint is None:
        # If setpoint is not available in config, use default setpoint to be 65

        temp_setpoint = weather_config.get("default_setpoint")

        logger.info("%s Default temperature setpoint %d", log_prefix('OfficeGoer'), temp_setpoint)

    input_cdd_vals = np.maximum(0, input_temp_vals - temp_setpoint)

    input_hdd_vals = np.maximum(0, temp_setpoint - input_temp_vals)

    # Get Month, Year value for each row

    day_epoch_vals = input_data[:, Cgbdisagg.INPUT_DAY_IDX]

    day_datetime_vals = pd.to_datetime(day_epoch_vals, unit='s')

    # Get year, month value in YYYYMM format
    month_vals = np.array((100 * day_datetime_vals.year + day_datetime_vals.month)).astype(int)

    month_vals_unique, month_vals_inv_idx, month_vals_counts = np.unique(month_vals, return_inverse=True,
                                                                         return_counts=True)
    logger.debug("%s Months: %s", log_prefix('Generic'), str(list(month_vals_unique)))

    # Get Day counts in all months

    month_day_vals = pd.DataFrame(np.array([month_vals, day_epoch_vals]).T,
                                  columns=['month', 'day']).drop_duplicates().values.astype(int)

    month_day_unique_vals, month_day_counts = np.unique(month_day_vals[:, 0], return_counts=True)

    # Reorder Month Day count rows based on month_vals_unique

    month_day_counts = \
        np.array([month_day_counts[month_day_unique_vals == month_val][0] for month_val in month_vals_unique])

    # Initialize seasonal arr and Index HDD, CDD, Temperature and seasonal values at month level

    num_season_val_rows = month_vals_unique.shape[0]

    seasonal_vals = np.empty((num_season_val_rows, 5))

    month_idx = raw_data_config.get("MONTH_IDX")
    temp_idx = raw_data_config.get("TEMP_IDX")
    hdd_idx = raw_data_config.get("HDD_IDX")
    cdd_idx = raw_data_config.get("CDD_IDX")
    season_idx = raw_data_config.get("SEASON_IDX")

    # get abs temperature diff(from setpoint) at monthly level

    seasonal_vals[:, month_idx] = month_vals_unique

    seasonal_vals[:, temp_idx] = np.bincount(month_vals_inv_idx, weights=input_temp_vals) / month_vals_counts

    seasonal_vals[:, temp_idx] = np.abs(seasonal_vals[:, temp_idx] - temp_setpoint)

    # get HDD/CDD vals at month level average over # days in month

    seasonal_vals[:, hdd_idx] = np.bincount(month_vals_inv_idx, weights=input_hdd_vals) / month_day_counts

    seasonal_vals[:, cdd_idx] = np.bincount(month_vals_inv_idx, weights=input_cdd_vals) / month_day_counts

    # initialisation seasonal value for each month value as summer season

    seasonal_vals[:, season_idx] = seasons.summer.value

    # get index for transition months based on max allowed transition month from config

    num_allowed_transition_months = int(num_season_val_rows * weather_config.get('transition_month_perc'))

    transition_months_row_idx = seasonal_vals[:, temp_idx].argsort()[:num_allowed_transition_months]

    seasonal_vals[transition_months_row_idx, season_idx] = seasons.transition.value

    # Get winter season months based on HDD/CDD values based on config

    winter_months_row_idx = (seasonal_vals[:, season_idx] == seasons.summer.value) & \
                            (seasonal_vals[:, hdd_idx] > seasonal_vals[:, cdd_idx])

    seasonal_vals[winter_months_row_idx, season_idx] = seasons.winter.value

    # convert transition months to winter months based on CDD/HDD values based on config

    t2s_cdd_hdd_diff = weather_config.get('t2s_cdd_hdd_diff')
    t2s_hdd = weather_config.get('t2s_hdd')

    transition_winter_row_idx = (seasonal_vals[:, season_idx] == seasons.transition.value) & \
                                (seasonal_vals[:, hdd_idx] > (seasonal_vals[:, cdd_idx] + t2s_cdd_hdd_diff)) & \
                                (seasonal_vals[:, cdd_idx] < t2s_hdd)

    seasonal_vals[transition_winter_row_idx, season_idx] = seasons.winter.value

    # convert transition months to summer months based on CDD/HDD values based on config

    transition_summer_row_idx = (seasonal_vals[:, season_idx] == seasons.transition.value) & \
                                (seasonal_vals[:, cdd_idx] > (seasonal_vals[:, hdd_idx] + t2s_cdd_hdd_diff)) & \
                                (seasonal_vals[:, hdd_idx] < t2s_hdd)

    seasonal_vals[transition_summer_row_idx, season_idx] = seasons.summer.value

    # Merge seasonal values to all rows of input data

    input_season_vals = \
        np.array([seasonal_vals[month_vals_unique == month_val, season_idx][0] for month_val in month_vals])

    logger.debug("%s Season Value Matrix: %s", log_prefix('Generic'), str(seasonal_vals.tolist()))

    # get index-map for monthly_season vals

    index_dict = {
        'MONTH_IDX' : month_idx,
        'TEMP_IDX'  : temp_idx,
        'HDD_IDX'   : hdd_idx,
        'CDD_IDX'   : cdd_idx,
        'SEASON_IDX': season_idx
    }

    t_seasonal_info_end = datetime.now()

    logger.info("%s Get seasonal information for lifestyle input took | %.3f s", log_prefix('Generic'),
                get_time_diff(t_seasonal_info_start, t_seasonal_info_end))

    return input_season_vals, seasonal_vals, index_dict


def get_cooling_estimate(lifestyle_input_object, disagg_input_object, disagg_output_object, logger_pass):

    """
    Parameters:
        lifestyle_input_object(dict)            : Dictionary containing lifestyle specific input
        disagg_input_object (dict)              : Dictionary containing all disagg inputs
        disagg_output_object(dict)              : Dictionary containing all disagg outputs
        logger_pass(dict)                       : Contains base logger and logging dictionary

    Returns:
        cooling_epoch_estimate(np.ndarray)      : 1-D Array for cooling estimates corresponding to input data
    """

    t_cooling_estimate_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_hvac_estimate')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Cooling Estimate for lifestyle input data", log_prefix('OfficeGoer'))

    # Retrieve lifestyle input data and module seq

    input_data = lifestyle_input_object.get('input_data')

    # initialize cooling estimate

    cooling_estimate = np.zeros(input_data.shape[0])

    # get raw hvac estimate from disagg output object

    module_seq = disagg_input_object.get('config').get('disagg_aer_seq')

    if 'hvac' in module_seq:

        # If hvac in modules run, then get cooling information from here

        cl_write_idx, ht_write_idx = disagg_output_object.get('output_write_idx_map').get('hvac')

        disagg_epoch_estimate = disagg_output_object.get('epoch_estimate', None)

        if disagg_epoch_estimate is not None:

            # Get cooling and heating epoch level estmates

            cl_epoch_estimate = disagg_epoch_estimate[:, cl_write_idx]

            logger.debug("%s Epoch Estimate available for cooling", log_prefix('OfficeGoer'))

        else:

            logger.warning("%s Epoch estimate not available for cooling", log_prefix('OfficeGoer'))

            return cooling_estimate

    else:

        logger.warning("%s hvac not available in module seq", log_prefix('OfficeGoer'))

        return cooling_estimate

    # get original input data from disagg_input object

    original_input_data = copy.deepcopy(lifestyle_input_object.get('original_input_data'))

    # replace actual consumption with cooling estimate for further preprocessing

    original_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = cl_epoch_estimate

    # get min and max days for lifestyle input data and trim cooling and heating data

    original_day_idx = original_input_data[:, Cgbdisagg.INPUT_DAY_IDX]

    min_day_idx = lifestyle_input_object.get("min_day_timestamp")

    max_day_idx = lifestyle_input_object.get("max_day_timestamp")

    cooling_raw_data = original_input_data[((original_day_idx >= min_day_idx) & (original_day_idx <= max_day_idx))]

    # downsample cooling to 1 hour rate

    cooling_raw_data = downsample_data(cooling_raw_data, Cgbdisagg.SEC_IN_HOUR)

    # Get cooling estimate from raw data

    cooling_estimate = cooling_raw_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    t_cooling_estimate_end = datetime.now()

    logger.info("%s Got cooling estimate in | %.3f s", log_prefix('OfficeGoer'),
                get_time_diff(t_cooling_estimate_start, t_cooling_estimate_end))

    return cooling_estimate
