"""
Author  -   Anand Kumar Singh
Date    -   22th Feb 2021
This file contains code for pre processing data and data preparation
"""

# Import python packages
import copy
import logging
import numpy as np

# Import functions from within the project

from python3.utils.find_runs import find_runs
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params
from python3.analytics.hvac_inefficiency.configs.pre_and_post_processing_config import get_pre_processing_config
from python3.analytics.hvac_inefficiency.configs.pre_and_post_processing_config import get_post_processing_config


def pre_inefficiency_sanity_checks(input_hvac_inefficiency_object, logger_pass, device):

    """
        This function estimates cycling based inefficiency

        Parameters:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            logger_pass                         (object)        logger object
            device                              (str)           string indicating device, either AC or SH
        Returns:
            run_inefficiency                    (Bool)          Whether or not to run inefficiency
    """

    # Initializing logger function

    logger_local = logger_pass.get("logger").getChild("pre_inefficiency_sanity_checks")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    logger.info('Pre processing checks for HVAC |')

    run_inefficiency = True

    device_consumption = input_hvac_inefficiency_object.get(device).get('demand_hvac_pivot').get('values')

    # Set NaN consumptions to zero

    nan_idx = np.isnan(device_consumption)
    device_consumption[nan_idx] = 0

    number_of_hvac_days = np.sum(device_consumption, axis=1)
    number_of_hvac_days = np.count_nonzero(number_of_hvac_days)

    hours_of_hvac_consumption = np.count_nonzero(device_consumption)

    pre_process_config = get_pre_processing_config()

    hvac_days_threshold = pre_process_config.get('hvac_days_threshold')
    hvac_hours_threshold = pre_process_config.get('hvac_hours_threshold')

    if number_of_hvac_days <= hvac_days_threshold:

        run_inefficiency = False
        logger.info('Not enough HVAC days, not running HVAC inefficiency | {}'.format(device))

    elif hours_of_hvac_consumption <= hvac_hours_threshold:

        run_inefficiency = False
        logger.info('Not enough HVAC hours, not running HVAC inefficiency | {}'.format(device))

    return run_inefficiency


def post_inefficiency_sanity_checks(input_inefficiency_object, output_inefficiency_object, logger_pass, device):

    """
        This function estimates cycling based inefficiency

        Parameters:
            input_inefficiency_object           (dict)          dictionary containing all input the information
            output_inefficiency_object          (dict)          dictionary containing all output information
            logger_pass                         (object)        logger object
            device                              (str)           string indicating device, either AC or SH
        Returns:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            output_hvac_inefficiency_object     (dict)          dictionary containing all output information
    """

    static_params = hvac_static_params()

    # Initializing logger function

    logger_local = logger_pass.get("logger").getChild("post_processing_hvac")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    logger.info('Post processing checks for HVAC inefficiency |')

    post_processing_config = get_post_processing_config()

    complete_mask = post_processing_config.get('complete_mask')

    sampling_rate = input_inefficiency_object.get('sampling_rate')
    fcc = output_inefficiency_object.get(device).get('cycling_debug_dictionary').get('full_cycle_consumption')
    cluster_count = output_inefficiency_object.get(device).get('updated_cluster_information').get('cluster_count')

    min_clusters_limit = post_processing_config.get('min_clusters_limit')
    fcc_15_min_limit = post_processing_config.get('fcc_15_min_limit')
    fcc_30_min_limit = post_processing_config.get('fcc_30_min_limit')

    if cluster_count < min_clusters_limit:
        logger.info('Cluster count = 1, masking all inefficiency |')
        complete_mask = True

    if (fcc <= fcc_15_min_limit) & (sampling_rate == Cgbdisagg.SEC_IN_15_MIN):
        logger.info('FCC value too small for {} sampling, masking all inefficiency'.format(sampling_rate))
        complete_mask = True
    elif (fcc <= fcc_30_min_limit) & (sampling_rate == Cgbdisagg.SEC_IN_30_MIN):
        logger.info('FCC value too small for {}, masking all inefficiency'.format(sampling_rate))
        complete_mask = True

    if complete_mask:
        output_inefficiency_object[device] = dict({})
        return input_inefficiency_object, output_inefficiency_object

    # Masking small short cycling count

    short_cycling = output_inefficiency_object.get(device).get('cycling_debug_dictionary').get('short_cycling')

    # Setting NaNs to 0
    nan_idx = np.isnan(short_cycling)
    short_cycling[nan_idx] = 0

    non_zero_idx = (short_cycling != 0)
    short_cycling[non_zero_idx] = 1

    run_values, run_starts, run_lengths = find_runs(short_cycling)
    valid_short_cycling = (run_values == 1)

    max_allowed_short_cycling_streaks = static_params.get('ineff').get('max_allowed_short_cycling_streaks')

    if valid_short_cycling.sum() <= max_allowed_short_cycling_streaks:
        short_cycling[:] = 0
        logger.info('Not enough short cycling found, masking short cycling |')
        output_inefficiency_object[device]['cycling_debug_dictionary']['short_cycling'] = copy.deepcopy(short_cycling)

    # Masking abrupt HVAC hours change less than 4 hours per day and non net consumption outliers

    minimum_hours_of_hvac = post_processing_config.get('minimum_hours_of_hvac')

    net_ao_consumption_days =\
        output_inefficiency_object.get(device).get('net_ao_outlier').get('final_outlier_days')

    net_consumption_days =\
        output_inefficiency_object.get(device).get('net_consumption_outlier').get('final_outlier_days')

    final_outlier_days =\
        copy.deepcopy(output_inefficiency_object.get(device).get('abrupt_hvac_hours').get('final_outlier_days'))

    if not final_outlier_days.shape[0] == 0:
        logger.debug('Removing less than {} hours amplitude outliers |'.format(minimum_hours_of_hvac))

        input_array = output_inefficiency_object.get(device).get('abrupt_hvac_hours').get('input_array')
        low_hours_index = input_array[:, 0] <= minimum_hours_of_hvac

        invalid_dates = input_array[low_hours_index, 1]
        invalid_dates = np.isin(final_outlier_days, invalid_dates)
        final_outlier_days = final_outlier_days[~invalid_dates]

        # Mask outliers that don't match with net consumption

        valid_dates = np.isin(final_outlier_days, net_consumption_days)
        final_outlier_days = final_outlier_days[valid_dates]

        output_inefficiency_object[device]['abrupt_hvac_hours']['final_outlier_days'] =\
            copy.deepcopy(final_outlier_days)

    # Masking abrupt HVAC hours change less than 4 hours per day

    minimum_hours_amplitude = post_processing_config.get('minimum_hours_amplitude')

    final_outlier_days =\
        copy.deepcopy(output_inefficiency_object.get(device).get('abrupt_amplitude').get('final_outlier_days'))

    if not final_outlier_days.shape[0] == 0:
        logger.debug('Removing less than {} Wh amplitude outliers |'.format(minimum_hours_amplitude))
        input_array = output_inefficiency_object.get(device).get('abrupt_amplitude').get('input_array')
        low_hours_index = input_array[:, 0] <= minimum_hours_of_hvac

        invalid_dates = input_array[low_hours_index, 1]
        invalid_dates = np.isin(final_outlier_days, invalid_dates)
        final_outlier_days = final_outlier_days[~invalid_dates]

        # Mask outliers that don't match with net consumption

        valid_dates = np.isin(final_outlier_days, net_consumption_days)
        final_outlier_days = final_outlier_days[valid_dates]

        output_inefficiency_object[device]['abrupt_amplitude']['final_outlier_days'] =\
            copy.deepcopy(final_outlier_days)

    # Remove AO outliers where there isn't overlapping with Net consumption AO

    final_outlier_days =\
        copy.deepcopy(output_inefficiency_object.get(device).get('abrupt_ao_hvac').get('final_outlier_days'))

    if not final_outlier_days.shape[0] == 0:

        # Mask outliers that don't match with net consumption

        valid_dates = np.isin(final_outlier_days, net_ao_consumption_days)
        final_outlier_days = final_outlier_days[valid_dates]

        output_inefficiency_object[device]['abrupt_ao_hvac']['final_outlier_days'] = copy.deepcopy(final_outlier_days)

    return input_inefficiency_object, output_inefficiency_object
