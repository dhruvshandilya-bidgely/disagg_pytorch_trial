"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module to remove missing data
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy
from scipy import interpolate

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def remove_missing_data(in_data, debug, ev_config, logger_base):
    """
    Function to remove missing data from input

        Parameters:
            in_data                   (np.ndarray)              : Input 21-column data
            logger_base               (logger)                  : logger base
            debug                     (dict)                    : Containing all important data/values as well as HSM
            ev_config                  (dict)                    : module config parameters

        Returns:
            input_data                (np.ndarray)              : Input 21-column data with missing data removed
            debug                     (dict)                    : Updated debug object

    """
    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('remove_missing_data')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    input_data = deepcopy(in_data)

    energy_idx = (input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0)

    # Get the daily peaks count

    unq_weeks, weeks_idx = np.unique(input_data[:, Cgbdisagg.INPUT_WEEK_IDX], return_inverse=True)
    weekly_count = np.bincount(weeks_idx, energy_idx)

    total_weeks_count = len(unq_weeks)

    zero_consumption_weeks = np.where(weekly_count == 0)[0]

    missing_points_idx = np.array([False] * input_data.shape[0])

    for idx in zero_consumption_weeks:
        data_idx = np.where(weeks_idx == idx)[0]

        missing_points_idx[data_idx] = True

    missing_data = input_data[missing_points_idx, :]

    input_data = input_data[~missing_points_idx, :]

    if len(zero_consumption_weeks) > 0:
        # Get seasonal distribution of missing data and overall data

        logger.info('Number of missing weeks | {}'.format(len(zero_consumption_weeks)))

        cutoff_temperature = ev_config['season_cutoff_temp']

        if np.nansum(missing_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]) == 0:
            points_per_week = Cgbdisagg.DAYS_IN_WEEK * Cgbdisagg.HRS_IN_DAY * debug['factor']

            temperature_data = in_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]

            weekly_avg_temperature = np.bincount(weeks_idx, temperature_data) / points_per_week

            missing_weeks_idx = np.where(np.isnan(weekly_avg_temperature))[0]

            valid_weeks_idx = np.where(~np.isnan(weekly_avg_temperature))[0]
            valid_weeks_temperature = weekly_avg_temperature[valid_weeks_idx]

            function_valid_weeks_idx = deepcopy(valid_weeks_idx)
            function_valid_weeks_temperature = deepcopy(valid_weeks_temperature)

            if missing_weeks_idx[0] == 0:
                function_valid_weeks_idx = np.r_[0, function_valid_weeks_idx]
                function_valid_weeks_temperature = np.r_[valid_weeks_temperature[0],
                                                         function_valid_weeks_temperature]

            if missing_weeks_idx[-1] == (total_weeks_count - 1):
                function_valid_weeks_idx = np.r_[function_valid_weeks_idx, total_weeks_count - 1]
                function_valid_weeks_temperature = np.r_[function_valid_weeks_temperature,
                                                         valid_weeks_temperature[-1]]

            temperature_func = interpolate.interp1d(function_valid_weeks_idx,
                                                    function_valid_weeks_temperature)

            weekly_avg_temperature = temperature_func(np.arange(0, total_weeks_count))

            valid_wtr_count = np.sum(weekly_avg_temperature[valid_weeks_idx] <= cutoff_temperature)
            missing_wtr_count = np.sum(weekly_avg_temperature[missing_weeks_idx] <= cutoff_temperature)

            valid_smr_count = np.sum(weekly_avg_temperature[valid_weeks_idx] > cutoff_temperature)
            missing_smr_count = np.sum(weekly_avg_temperature[missing_weeks_idx] > cutoff_temperature)

            wtr_missing_factor = (valid_wtr_count + missing_wtr_count) / valid_wtr_count
            smr_missing_factor = (valid_smr_count + missing_smr_count) / valid_smr_count
        else:
            valid_wtr_count = np.sum(input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX] <= cutoff_temperature)
            missing_wtr_count = np.sum(missing_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX] <= cutoff_temperature)

            valid_smr_count = np.sum(input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX] > cutoff_temperature)
            missing_smr_count = np.sum(missing_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX] > cutoff_temperature)

            wtr_missing_factor = (valid_wtr_count + missing_wtr_count) / valid_wtr_count
            smr_missing_factor = (valid_smr_count + missing_smr_count) / valid_smr_count
    else:
        wtr_missing_factor = 1
        smr_missing_factor = 1

    # Handling NaN for missing seasons

    if np.isnan(wtr_missing_factor):
        wtr_missing_factor = 1
    if np.isnan(smr_missing_factor):
        smr_missing_factor = 1

    logger.info('Missing factor for winter | {}'.format(wtr_missing_factor))
    logger.info('Missing factor for summer | {}'.format(smr_missing_factor))

    debug['missing_data_info'] = {
        'wtr_factor': wtr_missing_factor,
        'smr_factor': smr_missing_factor
    }

    debug['valid_days_count'] = len(np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX]))

    debug['input_data_missing'] = deepcopy(missing_data)

    debug['processed_input_data'] = deepcopy(input_data)

    return input_data, debug
