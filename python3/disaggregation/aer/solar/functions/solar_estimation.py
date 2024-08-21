"""
Author - Anand Kumar Singh
Date - 14th Feb 2020
This file has code for run solar estimation for a user

"""

# Import python packages

import logging
import datetime
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff
from python3.disaggregation.aer.solar.functions.curve_estimation import estimate_normalised_curve
from python3.disaggregation.aer.solar.functions.estimate_capacity_negative_consumption import estimate_solar_capacity_neg_cons


def solar_smoothening(input_data, nightload_array, solar_generation_data, valid_day_array, predicted_capacity):

    """
        This function smoothens solar generation for users with

        Parameters:
            input_data             (numpy.ndarray)     numpy array containing 21 column matrix and solar potential
            nightload_array        (numpy.ndarray)     array for nightload values for each day
            solar_generation_data  (numpy.ndarray)     solar generation data for the user
            valid_day_array        (numpy.ndarray)     array containing valid day information
            predicted_capacity     (float)             predicted capacity for the user

        Returns:
            solar_generation_data  (numpy.ndarray)     smoothened solar generation data for the user
    """

    # Sunlight present when time is between sunrise and sunset times
    sun_presence = np.logical_and(
        input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] >= input_data[:, Cgbdisagg.INPUT_SUNRISE_IDX],
        input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] <= input_data[:, Cgbdisagg.INPUT_SUNSET_IDX])

    nightload_array = np.array(nightload_array)
    nightload_array[nightload_array<0] = 0
    positive_idx = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] + solar_generation_data >= nightload_array
    positive_idx_non_valid = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] + solar_generation_data >= 0

    # Wherever solar generation + consumption <0, add smoothening of 10%ile
    valid_day = 0
    for day in np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX]):

        day_idx = (input_data[:, Cgbdisagg.INPUT_DAY_IDX] == day) & (sun_presence)
        if len(input_data[day_idx])>0:
            valid_day = valid_day_array[day_idx][0]

        if valid_day:
            negative_idx = (~positive_idx) & (day_idx)
            minimum_consumption_idx = (day_idx) & (positive_idx)
        else:
            negative_idx = (~positive_idx_non_valid) & (day_idx)
            minimum_consumption_idx = (day_idx) & (positive_idx_non_valid)

        if len(input_data[minimum_consumption_idx]) > 0:
            daily_minimum_consumption = np.nanquantile(
                input_data[minimum_consumption_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] + solar_generation_data[
                    minimum_consumption_idx], 0.1)

            solar_generation_data[negative_idx] = min(daily_minimum_consumption, predicted_capacity) + \
                                                  abs(input_data[negative_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # Replace values by nightload where generation+consumption still < 0
    negative_idx_non_valid = (input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] + solar_generation_data <= 0) & (sun_presence)
    solar_generation_data[negative_idx_non_valid] = abs(input_data[negative_idx_non_valid, Cgbdisagg.INPUT_CONSUMPTION_IDX]) + nightload_array[negative_idx_non_valid]

    nan_idx_generation = np.isnan(solar_generation_data)
    solar_generation_data[nan_idx_generation] = 0
    # Maximum generation at any point should be less than predicted capacity
    solar_generation_data[solar_generation_data > predicted_capacity] = predicted_capacity

    return solar_generation_data


def estimate_solar_consumption(input_data, solar_config, irradiance, logger_pass, detection_hsm):

    """
        This function estimates solar generation for users with

        Parameters:
            input_data             (numpy.ndarray)     numpy array containing 21 column matrix and solar potential
            solar_config           (dict)              r_squared threshold for good days
            logger_pass            (object)            logger object
            irradiance             (numpy.ndarray)     array containing irradiance values
            detection_hsm          (list)              Solar detection hsm
        Returns:
            input_data             (float)             numpy array containing 21 column matrix and solar generation
            estimation_hsm         (dict)              Estimation HSM dictionary
    """


    # Taking new logger base for this module
    logger_local = logger_pass.get("logger").getChild("estimate_solar_consumption")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    # Curve estimation
    start_time = datetime.datetime.now()

    # Estimate solar potential for each data point
    logger.debug('Estimate solar potential for each data point |')
    input_data, input_features_array, curve_hsm, debug_dictionary_curve =\
        estimate_normalised_curve(input_data, solar_config, irradiance, logger_pass)

    logger.info('Successfully estimated solar generation curve |')


    end_time = datetime.datetime.now()
    curve_estimation_time = get_time_diff(start_time, end_time)
    logger.info('Timing: Solar curve estimation took | %0.3f', curve_estimation_time)

    # Capacity estimation based on data type
    start_time = datetime.datetime.now()

    # Initialise empty hsm
    capacity_hsm = {}

    # Estimate Capacity based on data type

    logger.debug('Estimate capacity based on data type |')
    if solar_config.get('estimation_data_type') == 'neg_consumption':
        logger.info('Start capacity estimation using negative data |')
        predicted_capacity, capacity_hsm, valid_day_array, debug_predicted_capacity, nightload_array = \
            estimate_solar_capacity_neg_cons(input_data, solar_config, detection_hsm, logger_pass)
        if solar_config.get('previous_capacity'):
            predicted_capacity = [solar_config.get('previous_capacity')]
    else:
        # Setting Default for capacity and debug
        logger.info('Unknown estimation data type, skipping capacity estimation |')
        predicted_capacity = 0
        debug_predicted_capacity = {}
        valid_day_array = np.ones_like(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX])
        nightload_array = np.ones_like(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX])

    predicted_capacity = predicted_capacity[0]

    remove_solar_gen = 0

    if predicted_capacity < 0:
        logger.info('Estimated Solar capacity is negative, skipping solar generation estimation | %0.3f', predicted_capacity)
        predicted_capacity = 0
        capacity_hsm['capacity'] = predicted_capacity
        remove_solar_gen = 1

    logger.info('Solar capacity estimation complete, capacity | %0.3f', predicted_capacity)
    end_time = datetime.datetime.now()
    capacity_estimation_time = get_time_diff(start_time, end_time)
    logger.info('Timing: Solar capacity estimation took | %0.3f', capacity_estimation_time)

    # Updating solar generation by accounting for capacity, NaNs and type 2 vacation
    logger.debug('Updating solar generation by accounting for capacity, NaNs and type 2 vacation |')

    solar_potential_column_idx = solar_config.get('solar_potential_column')
    nan_idx = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] == np.nan
    not_nan_idx = ~ (nan_idx)
    factor = 1
    solar_generation_data =\
        input_data[:, solar_potential_column_idx] * predicted_capacity * valid_day_array * not_nan_idx * factor

    # Run smoothening algorithm on solar generation data
    logger.debug('Running solar smoothening |')
    solar_generation_data = solar_smoothening(input_data, nightload_array, solar_generation_data, valid_day_array, predicted_capacity)

    input_data = np.c_[input_data, solar_generation_data]
    solar_generation = np.nansum(solar_generation_data)

    #output bill cycles
    logger.debug('Calculating index for data points in out_bill_cycles |')
    out_bill_cycles = solar_config.get('out_bill_cycles')
    if type(out_bill_cycles) == np.ndarray:
        out_bill_cycles_idx = np.isin(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], out_bill_cycles)
    else:
        out_bill_cycles_idx = np.ones_like(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX], dtype=bool)

    # Start/End Date given by first/last occurence of valid day
    valid_day_array = valid_day_array * out_bill_cycles_idx
    start_date = int(input_data[np.argmax(valid_day_array > 0), Cgbdisagg.INPUT_EPOCH_IDX])
    end_date = int(input_data[len(input_data) - np.argmax(valid_day_array[::-1] > 0) - 1, Cgbdisagg.INPUT_EPOCH_IDX])

    logger.info('Solar Generation Start Date |' + str(start_date))
    logger.info('Solar Generation End Date |' + str(end_date))

    # capacity estimation based on data type
    estimation_hsm = {**capacity_hsm, **curve_hsm}
    estimation_hsm['solar_generation'] = solar_generation
    estimation_hsm['start_date'] = start_date
    estimation_hsm['end_date'] = end_date
    estimation_hsm['remove_solar_gen'] = remove_solar_gen

    return input_data, estimation_hsm
