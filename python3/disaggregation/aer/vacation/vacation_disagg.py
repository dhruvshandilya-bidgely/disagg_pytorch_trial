"""
Author - Mayank Sharan
Date - 27/8/19
Run different components of vacation disaggregation
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# Import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff

from python3.disaggregation.aer.vacation.functions.label_type_1 import label_type_1
from python3.disaggregation.aer.vacation.functions.label_type_2 import label_type_2
from python3.disaggregation.aer.vacation.functions.get_day_data import get_day_data
from python3.disaggregation.aer.vacation.functions.compute_day_power import compute_day_power
from python3.disaggregation.aer.vacation.functions.mask_timed_devices import mask_timed_devices
from python3.disaggregation.aer.vacation.functions.compute_day_baseload import compute_day_baseload
from python3.disaggregation.aer.vacation.functions.remove_lap_thin_pulses import remove_lap_thin_pulses
from python3.disaggregation.aer.vacation.functions.prepare_vacation_output import prepare_vacation_output
from python3.disaggregation.aer.vacation.functions.identify_lap_thin_pulses import identify_lap_thin_pulses


def vacation_disagg(input_data, vacation_config, timed_disagg_output, logger_pass):

    """
    Run the vacation disaggregation module

    Parameters:
        input_data          (np.ndarray)        : 21 column input data
        vacation_config     (dict)              : Dictionary containing all needed configuration variables
        timed_disagg_output (dict)              : Contains all timed devices' consumption like pp and twh
        logger_pass         (dict)              : Contains the logger and the logging dictionary to be passed on

    Returns:
        debug               (dict)              : Contains all variables needed for debugging
        type_1_epoch        (np.ndarray)        : Array marking type 1 vacation at epoch level
        type_2_epoch        (np.ndarray)        : Array marking type 2 vacation at epoch level
    """

    t_vac_start = datetime.now()

    # ------------------------------------------ STAGE 1: INITIALISATIONS ---------------------------------------------

    t_init_start = datetime.now()

    # Initialize vacation logger

    logger_base = logger_pass.get('base_logger').getChild('vacation_disagg')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Prepare logger pass to pass to sub-modules

    logger_pass['base_logger'] = logger_base

    # Initialize an empty debug dictionary

    debug = {}

    # Log the type of vacation config we are using for the user

    if vacation_config.get('user_info').get('is_europe'):
        logger.info('Type of vacation config initialized | Europe')
    else:
        logger.info('Type of vacation config initialized | Default')

    t_init_end = datetime.now()

    logger.debug('Initialization took | %.3f s ', get_time_diff(t_init_start, t_init_end))

    # ------------------------------------------ STAGE 2: DATA PREPARATION --------------------------------------------

    t_data_prep_start = datetime.now()

    # Mask timed devices' consumption from the input signal

    input_data_masked, valid_mask_cons_bool = mask_timed_devices(input_data, timed_disagg_output, logger_pass)

    # Identify thin pulses that lie in LAPs

    peak_amp_arr, peak_removal_bool = identify_lap_thin_pulses(input_data_masked, vacation_config, logger_pass)

    # Convert 1d data to 2d day wise matrices for further processing

    return_dict = get_day_data(input_data, input_data_masked, valid_mask_cons_bool, peak_amp_arr, peak_removal_bool,
                               vacation_config, logger_pass)

    day_ts = return_dict.get('day_ts')
    day_data = return_dict.get('day_data')
    month_ts = return_dict.get('month_ts')
    epoch_ts = return_dict.get('epoch_ts')
    day_data_masked = return_dict.get('day_data_masked')
    day_peak_amp_arr = return_dict.get('day_peak_amp_arr')
    day_peak_removal = return_dict.get('day_peak_removal')
    day_valid_mask_cons = return_dict.get('day_valid_mask_cons')

    # Remove LAP peaks from consumption data

    day_data_processed, vacation_config = remove_lap_thin_pulses(day_data_masked, day_peak_removal, day_peak_amp_arr,
                                                                 vacation_config, logger_pass)

    # Compute day level count of nan values

    day_nan_count = np.sum(np.isnan(day_data_masked), axis=1) - np.sum(day_valid_mask_cons, axis=1)

    # Populate the debug dictionary for data preparation step

    debug['data'] = {
        'day_ts': day_ts,
        'day_data': day_data,
        'month_ts': month_ts,
        'epoch_ts': epoch_ts,
        'input_data': input_data,
        'day_nan_count': day_nan_count,
        'day_data_masked': day_data_masked,
        'input_data_masked': input_data_masked,
        'day_data_processed': day_data_processed,
        'day_valid_mask_cons': day_valid_mask_cons,
    }

    t_data_prep_end = datetime.now()

    logger.debug('Data preparation took | %.3f s ', get_time_diff(t_data_prep_start, t_data_prep_end))

    # ------------------------------------------ STAGE 3: FEATURE COMPUTATION -----------------------------------------

    t_feature_comp_start = datetime.now()

    # Compute power attributed to each day

    day_wise_power = compute_day_power(day_data_processed, day_nan_count, vacation_config)

    # Compute baseload attributed to each day

    day_wise_baseload = compute_day_baseload(day_data_processed, vacation_config)

    # Populate the debug dictionary for feature computation step

    debug['features'] = {
        'day_wise_power': day_wise_power,
        'day_wise_baseload': day_wise_baseload,
    }

    t_feature_comp_end = datetime.now()

    logger.debug('Feature computation took | %.3f s ', get_time_diff(t_feature_comp_start, t_feature_comp_end))

    # -------------------------------------------- STAGE 4: VACATION LABELING -----------------------------------------

    t_vac_labeling_start = datetime.now()

    vac_label = np.zeros_like(day_wise_baseload)
    vac_confidence = np.full(shape=day_wise_baseload.shape, fill_value=np.nan)

    # Mark type 1 vacation days

    type_1_bool, debug, vac_confidence = label_type_1(day_data_processed, day_wise_power, day_wise_baseload,
                                                      day_valid_mask_cons, vac_confidence, debug, vacation_config,
                                                      logger_pass)
    vac_label[type_1_bool] = 1

    # Mark type 2 vacation days

    type_2_bool, vac_confidence = label_type_2(day_data, day_nan_count, vac_confidence, vacation_config, logger_pass)
    vac_label[type_2_bool] = 2

    # Populate the debug dictionary for labeling step

    debug['labeling']['label'] = vac_label
    debug['labeling']['confidence'] = vac_confidence

    t_vac_labeling_end = datetime.now()

    logger.debug('Vacation labeling took | %.3f s ', get_time_diff(t_vac_labeling_start, t_vac_labeling_end))

    # ------------------------------------------- STAGE 5: OUTPUT PREPARATION -----------------------------------------

    t_output_prep_start = datetime.now()

    debug, type_1_epoch, type_2_epoch = prepare_vacation_output(vac_label, day_ts, epoch_ts, debug, vacation_config)

    t_output_prep_end = datetime.now()

    logger.debug('Vacation output prep took | %.3f s ', get_time_diff(t_output_prep_start, t_output_prep_end))

    t_vac_end = datetime.now()

    logger.info('Time taken to run vacation is | %.3f s ', get_time_diff(t_vac_start, t_vac_end))

    return debug, type_1_epoch, type_2_epoch
