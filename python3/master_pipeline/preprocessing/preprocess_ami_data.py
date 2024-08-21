"""
Author - Mayank Sharan
Date - 24/09/18
Pre-processes input data using different utility functions for disagg modules to use
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff
from python3.utils.log_data_attributes import log_data_attributes

from python3.master_pipeline.preprocessing.downsample_data import downsample_data
from python3.master_pipeline.preprocessing.check_data_quality import check_data_quality
from python3.master_pipeline.preprocessing.spread_hourly_columns import spread_hourly_columns
from python3.master_pipeline.preprocessing.enforce_column_sanity import enforce_column_sanity
from python3.master_pipeline.preprocessing.modify_high_consumption import modify_high_consumption
from python3.master_pipeline.preprocessing.interpolate_temperature import interpolate_temperature
from python3.master_pipeline.preprocessing.homogenise_data_sampling_rate import homogenise_data_sampling_rate


def preprocess_ami_electric(pipeline_input_object):

    """
    Pre process raw disagg input data dictionary to prepare it for consumption by the disagg pipeline

    Parameters:
        pipeline_input_object (dict)              : Dictionary containing all inputs

    Returns:
        pipeline_input_object (dict)              : Dictionary containing all inputs
    """

    # Initialise logger for the run

    logger_base = pipeline_input_object['logger'].getChild('preprocess_ami_electric')
    logger = logging.LoggerAdapter(logger_base, pipeline_input_object.get('logging_dict'))

    # Initialize logger pass to send to functions inside

    logger_pass = {
        'base_logger': logger_base,
        'logging_dict': pipeline_input_object.get('logging_dict')
    }

    # Code to evaluate data quality and a gatekeeper decision to run the pipeline or not

    t_before_check = datetime.now()
    pipeline_input_object, run_pipeline, rejection_reasons = check_data_quality(pipeline_input_object, logger_base)
    t_after_check = datetime.now()

    logger.info('Data quality check took | %.3f s', get_time_diff(t_before_check, t_after_check))

    # Populate and return data check values

    data_quality_dict = {
        'run_pipeline': run_pipeline,
        'rejection_reasons': rejection_reasons,
    }

    if not run_pipeline:
        logger.info('Pipeline will not be run for the following reasons | %s', '; '.join(rejection_reasons))

        # Putting a default sampling rate so that post disagg results doesn't break

        pipeline_input_object['global_config']['sampling_rate'] = Cgbdisagg.SEC_IN_15_MIN
        pipeline_input_object['data_quality_metrics']['disagg_data_quality'] = data_quality_dict
        return pipeline_input_object

    # Populate original input data

    input_data = pipeline_input_object['input_data']
    pipeline_input_object['original_input_data'] = copy.deepcopy(input_data)

    # Copy hvac data points in the last 6 columns to each epoch value

    t_before_spread = datetime.now()
    input_data = spread_hourly_columns(input_data)
    t_after_spread = datetime.now()

    logger.info('Hourly spread columns took | %.3f s', get_time_diff(t_before_spread, t_after_spread))

    # Homogenize data by sampling rate

    t_before_homogenization = datetime.now()

    downsample_rate = pipeline_input_object.get('global_config').get('downsample_rate')

    if downsample_rate is None:
        sampling_rate, input_data = homogenise_data_sampling_rate(input_data)
    else:
        logger.info('Downsampling rate override | %.1f', downsample_rate)
        sampling_rate, input_data = homogenise_data_sampling_rate(input_data, downsample_rate)

    t_after_homogenization = datetime.now()

    # Make sure the data going into the pipeline is at least 15 minutes apart by sampling

    if sampling_rate < Cgbdisagg.SEC_IN_15_MIN and Cgbdisagg.SEC_IN_15_MIN % sampling_rate == 0:
        sampling_rate = Cgbdisagg.SEC_IN_15_MIN
        input_data = downsample_data(input_data, sampling_rate)

    logger.info('Homogenise data took | %.3f s', get_time_diff(t_before_homogenization, t_after_homogenization))

    pipeline_input_object['global_config']['sampling_rate'] = sampling_rate
    logger.info('Sampling rate for the given data is | %.1f', sampling_rate)

    if sampling_rate > Cgbdisagg.SEC_IN_HOUR or not(sampling_rate % Cgbdisagg.SEC_IN_15_MIN == 0):

        logger.warning('Pipeline will not be run for the following reasons | Invalid sampling rate')

        run_pipeline = False

        rejection_reasons.append('Invalid sampling rate')

        data_quality_dict = {
            'run_pipeline': run_pipeline,
            'rejection_reasons': rejection_reasons,
        }

        pipeline_input_object['data_quality_metrics']['disagg_data_quality'] = data_quality_dict

        return pipeline_input_object

    # We perform temperature interpolation if the temperature is present for less than 90% of the data

    if pipeline_input_object['data_quality_metrics']['present_temp_perc'] < Cgbdisagg.MIN_TEMP_NOT_INTERPOLATE:
        t_before_temp_interpolation = datetime.now()
        input_data = interpolate_temperature(sampling_rate, input_data)
        t_after_temp_interpolation = datetime.now()

        logger.info('Temperature interpolation took | %.3f s', get_time_diff(t_before_temp_interpolation,
                                                                             t_after_temp_interpolation))
    else:
        logger.info('Bypassing temperature interpolation since requirement satisfied |')

    # Set valid bool arrays

    is_valid_temp = np.logical_not(np.isnan(input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]))
    pipeline_input_object['data_quality_metrics']['is_valid_temp'] = is_valid_temp

    is_nan_cons = np.isnan(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    is_zero_cons = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] == 0
    is_neg_cons = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] < 0

    pipeline_input_object['data_quality_metrics']['is_nan_cons'] = is_nan_cons
    pipeline_input_object['data_quality_metrics']['is_neg_cons'] = is_neg_cons

    pipeline_input_object['data_quality_metrics']['is_valid_cons'] = np.logical_not(np.logical_or(is_nan_cons,
                                                                                                  is_zero_cons))

    # creating data copy without negative values
    pipeline_input_object['input_data_with_neg_and_nan'] = copy.deepcopy(input_data)

    bill_cycles, inv_idx, pts_per_bill_cycle = np.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                                                         return_inverse=True, return_counts=True)

    bc_consumption = np.bincount(inv_idx, weights=np.nan_to_num(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))
    total_cons_log = [(int(bill_cycles[i]), bc_consumption[i]) for i in range(bill_cycles.shape[0])]

    logger.info('Total Consumption (in Wh) by billing cycle (without neg and nan removal) are | %s', str(total_cons_log).replace('\n', ' '))

    # Set nan and negative consumption values to 0

    input_data[is_nan_cons, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0
    input_data[is_neg_cons, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    # saving and logging the total consumption before outlier removal

    pipeline_input_object['input_data_without_outlier_removal'] = copy.deepcopy(input_data)

    bill_cycles, inv_idx, pts_per_bill_cycle = np.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                                                         return_inverse=True, return_counts=True)

    bc_consumption = np.bincount(inv_idx, weights=input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    total_cons_log = [(int(bill_cycles[i]), bc_consumption[i]) for i in range(bill_cycles.shape[0])]

    logger.info('Total Consumption (in Wh) by billing cycle (without outlier removal) are | %s', str(total_cons_log).replace('\n', ' '))


    # Handle all high consumption values

    t_before_cons_modification = datetime.now()
    input_data = modify_high_consumption(input_data, sampling_rate, logger_pass)
    t_after_cons_modification = datetime.now()

    logger.info('High consumption modification took | %.3f s', get_time_diff(t_before_cons_modification,
                                                                             t_after_cons_modification))

    # Apply per column sanity checks

    input_data = enforce_column_sanity(input_data, logger_pass)

    # Set the modified input data to disagg input object

    pipeline_input_object['input_data'] = input_data
    pipeline_input_object['data_quality_metrics']['disagg_data_quality'] = data_quality_dict

    # Call a function to log attributes of the data

    t_before_data_log = datetime.now()
    log_data_attributes(pipeline_input_object, logger_pass)
    t_after_data_log = datetime.now()

    logger.info('Logging data features took | %.3f s', get_time_diff(t_before_data_log, t_after_data_log))

    return pipeline_input_object
