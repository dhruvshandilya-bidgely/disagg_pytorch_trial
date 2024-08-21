"""
Author - Sahana M
Date - 16th September 2021
This function is used as an alternative weather data analytics module incase the previous module fails
"""

# Import python packages
import logging
import numpy as np

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.itemization.aer.water_heater.functions.math_utils import find_seq


def get_2d_data(input_data, wh_config):
    """
    This function converts the epoch level consumption column into a 2D matrix
    Parameters:
        input_data          (ndarray)           : Input 21 column array
        wh_config           (dict)              : WH configurations dictionary

    Returns:
        row_idx             (np.ndarray)        : Epoch to Day level mapping
        col_idx             (np.ndarray)        : Epoch to Time mapping
    """

    # Extract the required variables

    sampling_rate = wh_config.get('user_info').get('sampling_rate')
    pd_mult = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    # Get the Epoch to Day level mapping

    day_ts, row_idx = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)

    # Get the Epoch to Time mapping

    col_idx = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] - input_data[:, Cgbdisagg.INPUT_DAY_IDX]
    col_idx = col_idx / Cgbdisagg.SEC_IN_HOUR
    col_idx = (pd_mult * (col_idx - col_idx.astype(int) + input_data[:, Cgbdisagg.INPUT_HOD_IDX])).astype(int)

    return row_idx, col_idx


def get_max_winter_temp(mean_daily_data, s_label):
    """
    This function is used to get the MAximum winter threshold
    Parameters:
        mean_daily_data          (np.ndarray)        : Input data
        s_label                  (np.ndarray)        : Season label
    Returns:
        max_winter_temp          (float)             : Maximum winter temperature
    """

    winter_days = s_label == -1
    if np.sum(winter_days):
        max_winter_temp = np.nanmax(mean_daily_data[winter_days])
    else:
        max_winter_temp = 'NA'

    return max_winter_temp


def get_max_tr_temp(mean_daily_data, s_label):
    """
    This function is used to get the Minimum cooling threshold
    Parameters:
        mean_daily_data          (np.ndarray)        : Input data
        s_label                  (np.ndarray)        : Season label
    Returns:
        cooling_min_thr          (float)             : Minimum threshold for cooling to start
    """

    trn_days = s_label == 0
    if np.sum(trn_days):
        max_trn_temp = np.nanmax(mean_daily_data[trn_days])
    else:
        max_trn_temp = 'NA'

    return max_trn_temp


def get_valid_season(s_label):
    """
    This function is used to get the number of seasons present
    Parameters:
        s_label             (np.ndarray)        : Season label
    Returns:
        valid seson bool    (np.ndarray)        : valid season
    """

    valid_season_bool = np.full(shape=3, fill_value=False)

    if -1 in s_label:
        valid_season_bool[0] = True

    if 1 in s_label:
        valid_season_bool[1] = True

    if 0 in s_label:
        valid_season_bool[2] = True

    return valid_season_bool


def get_s_label(input_data):
    """
    This function is used to get the season label from input data
    Parameters:
        input_data          (np.ndarray)        : Input data
    Returns:
        s_label             (np.ndarray)        : Season label
    """

    # Get the number of unique days
    unique_days = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX])
    new_s_label = []

    # For each unique day get the season label
    for i in unique_days:
        days = input_data[:, Cgbdisagg.INPUT_DAY_IDX] == i
        s_label_day = np.nanmedian(input_data[days, Cgbdisagg.INPUT_S_LABEL_IDX])
        new_s_label.append(s_label_day)
    new_s_label = np.asarray(new_s_label)

    s_label = new_s_label

    return s_label


def get_cooling_potential(input_data):
    """
    This function is used to get the cooling potential from input data
    Parameters:
        input_data          (np.ndarray)        : Input data
    Returns:
        new_cooling_pot     (np.ndarray)        : Cooling potential
    """

    # Get the number of unique days
    unique_days = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX])

    # For each unique day get the cooling potential
    new_cooling_pot = []
    for i in unique_days:
        days = input_data[:, Cgbdisagg.INPUT_DAY_IDX] == i
        cooling_day = np.nanmedian(input_data[days, Cgbdisagg.INPUT_COOLING_POTENTIAL_IDX])
        new_cooling_pot.append(cooling_day)
    new_cooling_pot = np.asarray(new_cooling_pot)

    return new_cooling_pot


def get_cooling_min_thr(mean_daily_data, cooling_potential):
    """
    This function is used to get the Minimum cooling threshold
    Parameters:
        mean_daily_data          (np.ndarray)        : Input data
        cooling_potential        (np.ndarray)        : Cooling potential
    Returns:
        cooling_min_thr          (float)             : Minimum threshold for cooling to start
    """

    cooling_days = cooling_potential > 0

    if np.sum(cooling_days):
        cooling_min_thr = np.nanmin(mean_daily_data[cooling_days])
    else:
        cooling_min_thr = 'NA'

    return cooling_min_thr


def get_weather_features(input_data, wh_config, logger_base):
    """
    Alternate weather data analytics module
    Parameters:
        input_data          (ndarray)           : Input 21 column array
        wh_config           (dict)              : WH configurations dictionary
        logger_base         (logger)            : Logger passed
    Returns:
        weather_data_output (dict)              : Dictionary containing all the weather derived outputs
        exit_swh            (Boolean)           : Boolean containing the run status of the function
    """

    # Initialise logger for this module

    logger_local = logger_base.get('logger').getChild('alternate_weather_data_output')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    exit_swh = False

    try:

        # Initialise the required variables

        rows, cols = get_2d_data(input_data, wh_config)
        temperature_data = input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]

        day_wise_temp_data = np.full(shape=(len(np.unique(rows)), len(np.unique(cols))), fill_value=np.nan)
        day_wise_temp_data[rows, cols] = temperature_data
        mean_daily_data = np.nanmean(day_wise_temp_data, axis=1)

        # Get Feels like temperature converted to a 24 column matrix

        factor = int(Cgbdisagg.SEC_IN_HOUR/wh_config.get('user_info').get('sampling_rate'))
        day_wise_data = np.full(shape=(day_wise_temp_data.shape[0], Cgbdisagg.HRS_IN_DAY), fill_value=0.0)

        for i in range(0, day_wise_temp_data.shape[1], factor):
            j = int(i / factor)
            day_wise_data[:, j] = np.nanmean(day_wise_temp_data[:, i:i + factor], axis=1)

        # Get season label

        s_label = get_s_label(input_data)

        # Get sequence array

        seq_arr = find_seq(s_label, min_seq_length=5)

        # Get the maximum winter temperature

        max_winter_temp = get_max_winter_temp(mean_daily_data, s_label)

        # Get the maximum transition temperature

        max_tr_temp = get_max_tr_temp(mean_daily_data, s_label)

        # Valid season bool

        valid_season_bool = get_valid_season(s_label)

        # Cooling potential

        cooling_potential = get_cooling_potential(input_data)

        # Get cooling minimum threshold

        cooling_min_thr = get_cooling_min_thr(mean_daily_data, cooling_potential)

        # Store all the info

        weather_data_output = dict()
        weather_data_output['weather'] = {
            'day_wise_data': {
                'fl': day_wise_data,
            },
            'season_detection_dict': {
                's_label': s_label,
                'seq_arr': seq_arr,
                'max_winter_temp': max_winter_temp,
                'max_tr_temp': max_tr_temp,
                'model_info_dict': {
                    'valid_season_bool': valid_season_bool
                }
            },
            'hvac_potential_dict': {
                'cooling_pot': cooling_potential,
                'cooling_min_thr': cooling_min_thr
            }
        }

        logger.info('Max winter temperature | {}'.format(max_winter_temp))
        logger.info('Max transition temperature is | {}'.format(max_tr_temp))

    except (ValueError, IndexError, KeyError):
        exit_swh = True
        weather_data_output = {}
        logger.info('Could not obtain Weather Data Features | ')

    return weather_data_output, exit_swh
