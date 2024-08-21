"""
Author - Anand Kumar Singh / Paras Tehria
Date - 14th Feb 2020
This file has code for run solar disaggregation module for a user

"""
# Import python packages

import logging
import numpy as np
from copy import deepcopy
from scipy.stats import mode

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.solar.functions.save_output import save_output
from python3.master_pipeline.preprocessing.downsample_data import downsample_data
from python3.disaggregation.aer.solar.functions.detect_solar_cnn import get_prob_solar_detection
from python3.disaggregation.aer.solar.functions.prep_solar_data import get_preprocessed_data
from python3.disaggregation.aer.solar.functions.solar_estimation import estimate_solar_consumption
from python3.disaggregation.aer.solar.functions.detect_solar_presence import get_solar_presence
from python3.disaggregation.aer.solar.functions.detect_solar_presence import get_detection_metrics


def negative_suntime_ratio(input_data, solar_config, sun_index):

    """
    Calculate ratio of negative points during suntime

    Parameters:
        input_data              (numpy.ndarray)         : Numpy array containing 21 column matrix
        solar_config            (dict)                  : Dict with all config info required by solar module
        sun_index               (float)                 : Sun index for input array

    Returns:
        neg_suntime_ratio       (float)                 : Ratio of negative points during suntime
    """


    unique_days = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX])
    min_days = solar_config.get('so_min_data_req')
    month, neg_suntime_ratio = 0, 0

    while month + min_days < len(unique_days):
        start = unique_days[month]
        end = unique_days[month + min_days]
        day_mask = (input_data[:, Cgbdisagg.INPUT_DAY_IDX] >= start) & (input_data[:, Cgbdisagg.INPUT_DAY_IDX] < end)
        neg_suntime_ratio = max(neg_suntime_ratio,
                                np.sum(np.logical_and(input_data[day_mask, Cgbdisagg.INPUT_CONSUMPTION_IDX] < 0,
                                                      input_data[day_mask, sun_index]).astype(int)) / np.nansum(input_data[day_mask, sun_index]))

        month += Cgbdisagg.DAYS_IN_MONTH
    return neg_suntime_ratio


def run_solar_estimation(input_data, solar_config, irradiance, logger_solar_pass, solar_detection, detection_hsm):

    """
    Estimate solar generation

    Parameters:
        input_data              (numpy.ndarray)         : Numpy array containing 21 column matrix
        solar_config            (dict)                  : Dict with all config info required by solar module
        irradiance              (numpy.ndarray)         : Numpy array with irradiance values
        logger_solar_pass       (object)                : Logger object
        solar_detection         (boolean)               : Solar detection flag
        detection_hsm           (list)                  : Solar detection hsm

    Returns:
        input_data              (numpy.ndarray)         : Numpy array containing 21 column matrix and 2 solar columns
        monthly_output          (numpy.ndarray)         : Numpy array containing monthly solar estimation
        estimation_hsm          (dict)                  : Dictionary containing estimation hsm
    """

    # Taking new logger base for this module
    logger_local = logger_solar_pass.get("logger").getChild("run_solar_estimation")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_solar_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_solar_pass.get("logging_dict"))

    # Get bill cycles in the data
    bill_cycle_ts, bill_cycle_idx, points_count = np.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                                                            return_counts=True, return_inverse=True)

    # Estimate solar generation if solar detected
    if solar_detection:
        logger.info('Solar detected, running solar estimation module')
        input_data, estimation_hsm = \
            estimate_solar_consumption(input_data, solar_config, irradiance, logger_pass, detection_hsm)

        solar_generation_column = solar_config.get('solar_generation_column')

        # Create monthly output array
        monthly_output = np.bincount(bill_cycle_idx, input_data[:, solar_generation_column])
        monthly_output = np.c_[bill_cycle_ts, monthly_output]

        # Updating input consumption with new data
        input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = \
            input_data[:, solar_generation_column] + input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    else:
        logger.info('Solar not detected, skipping solar estimation')
        estimation_hsm = {}
        monthly_output = np.zeros_like(bill_cycle_ts)
        monthly_output = np.c_[bill_cycle_ts, monthly_output]

    return input_data, monthly_output, estimation_hsm


def run_solar_detection(global_config, data, disagg_input_object, solar_config, logger_base):

    """
    This is the main function that detects solar power generation and updates in the hsm

        Parameters:
            global_config         (dict)             : global config
            data                 (np.ndarray)       : input 21 column matrix
            disagg_input_object  (dict)             : disagg input object
            solar_config          (dict)             : config for solar detection
            logger_base          (dict)             : logger object to generate logs

        Return:
            hsm                  (dict)             : updated hsm of the user containing presence of solar generation
    """

    input_data = deepcopy(data)

    # Initializing new logger child solar_disagg

    logger_local = logger_base.get('logger').getChild('solar_detection')

    # Initializing new logger pass to be used by the internal functions of solar_disagg

    logger_pass = {'logger': logger_local, 'logging_dict': logger_base.get('logging_dict')}

    # logger

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    num_days = len(np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX]))

    write_hsm = True

    if num_days < solar_config.get('so_min_data_req'):
        write_hsm = False
        hsm = {
            'timestamp': int(input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX]),
            'attributes': {
                'confidence': 0.00,
                'solar_present': 0,
                'instance_probabilities': [-1.0],
                'start_date': None,
                'end_date': None,
                'kind': None
            }
        }
        return hsm, write_hsm

    # array contains one hot encoded array containing presence of sunlight
    sun_array = np.ones((len(input_data), 1), dtype=input_data.dtype)

    # Sunlight present when time is between sunrise and sunset times
    sun_presence = np.logical_and(
        input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] >= input_data[:, Cgbdisagg.INPUT_SUNRISE_IDX],
        input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] <= input_data[:, Cgbdisagg.INPUT_SUNSET_IDX])

    logger.debug('Created one-hot encoded sunlight presence array | ')

    # Adding the one-hot encoded sun_array to act as a new column
    sun_array[:, 0] = np.where(sun_presence, sun_array[:, 0], 0)
    nan_sunrise_sunset = np.logical_or(np.isnan(input_data[:, Cgbdisagg.INPUT_SUNRISE_IDX]),
                                       np.isnan(input_data[:, Cgbdisagg.INPUT_SUNSET_IDX]))
    sun_array[:, 0] = np.where(~nan_sunrise_sunset, sun_array[:, 0], np.nan)

    input_data = np.hstack((input_data, sun_array))
    logger.debug('sunlight presence array added to 21 column matrix | ')

    # Basic checks to detect solar
    sun_index = solar_config.get('prep_solar_data').get('sun_index')

    # Check for Negative suntime ratio
    neg_suntime_ratio = negative_suntime_ratio(input_data, solar_config, sun_index)

    # Pre-processing data to create instances for detection
    logger.debug('Preprocessing data | ')

    # Down sampling data to 60 min granularity
    downsampled_input_data = downsample_data(input_data=input_data, target_rate=Cgbdisagg.SEC_IN_HOUR)
    logger.info('Data downsampled successfully | ')

    chunk_times, cnn_detection_array = \
        get_preprocessed_data(input_data=downsampled_input_data, logger_base=logger_pass, solar_config=solar_config)

    logger.debug('Data preprocessed successfully | ')

    neg_suntime_thresh = solar_config.get('solar_disagg').get('neg_suntime_thresh')


    if neg_suntime_ratio >= neg_suntime_thresh:

        logger.info('A lot of negatives during sun time solar detected |  ')

        probability_solar = np.ones(len(cnn_detection_array)).astype(float)
        probability_solar_including_disconn = probability_solar
        confidence = np.round(np.mean(probability_solar), 2)
        solar_present = 1
        irradiance, start_date, end_date, kind = get_detection_metrics(solar_config, input_data, solar_present, logger_pass)

    else:
        # Getting the solar detection and corresponding confidence values

        max_consumption = np.nanquantile(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], .98)
        sampling_rate = int(mode(np.diff(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]))[0][0])
        max_consumption_threshold = 400 *sampling_rate / Cgbdisagg.SEC_IN_HOUR
        if (neg_suntime_ratio == 0) & (max_consumption < max_consumption_threshold):

            logger.info("Consumption is very low for the user, solar not detected")
            write_hsm = False
            hsm = {
                'timestamp': int(input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX]),
                'attributes': {
                    'confidence': 0.00,
                    'solar_present': 0,
                    'instance_probabilities': [-1.0],
                    'start_date': None,
                    'end_date': None,
                    'kind': None
                }
            }
            return hsm, write_hsm

        else:
            _, probability_solar, probability_solar_including_disconn = get_prob_solar_detection(
                detection_arr_original=cnn_detection_array,
                disagg_input_object=disagg_input_object,
                logger_base=logger_pass,
                solar_config=solar_config)

        # check if probability solar becomes empty after removing disconnections in post-processing

        non_disconn_ratio = solar_config.get('solar_disagg').get('non_disconn_ratio')

        if len(probability_solar) <= non_disconn_ratio * len(probability_solar_including_disconn):

            logger.info("High number of disconnection instances found, writing result instance probability as -2")

            confidence = 0.00
            solar_present = 0
            probability_solar_including_disconn = [-2.0]
            start_date = None
            end_date = None
            kind = None
            irradiance = np.zeros_like(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX])

        else:
            confidence_cnn = np.round(np.mean(probability_solar), 2)

            irradiance, solar_present, confidence, start_date, end_date, kind = get_solar_presence(
                input_array=input_data,
                disagg_input_object=disagg_input_object,
                confidence_cnn=confidence_cnn,
                logger_base=logger_pass,
                solar_config=solar_config)

    # creating a hsm dict containing solar presence info

    hsm = {
        'timestamp': int(input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX]),
        'attributes': {
            'confidence': np.round(confidence, 2),
            'solar_present': solar_present,
            'instance_probabilities': list(probability_solar_including_disconn),
            'chunk_start': chunk_times['chunk_start'],
            'chunk_end': chunk_times['chunk_end'],
            'irradiance': irradiance,
            'start_date': start_date,
            'end_date': end_date,
            'kind': kind
        }
    }

    out_bill_cycle = disagg_input_object.get('out_bill_cycles')
    min_out_bc = out_bill_cycle.min()

    data_post_min_bc = deepcopy(input_data)
    data_post_min_bc = data_post_min_bc[data_post_min_bc[:, Cgbdisagg.INPUT_EPOCH_IDX] > min_out_bc, :]

    # saving heatmaps for debug
    if len(data_post_min_bc)>Cgbdisagg.DAYS_IN_MONTH:
        save_output(global_config, disagg_input_object, solar_config, data_post_min_bc, confidence, probability_solar_including_disconn,
                    logger_pass, start_date, end_date, kind)
    else:
        logger.info("Not enough data points in the given billing cycles")

    return hsm, write_hsm
