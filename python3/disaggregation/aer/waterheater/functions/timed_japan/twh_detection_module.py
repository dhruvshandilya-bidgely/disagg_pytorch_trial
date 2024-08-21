"""
Author - Sahana M
Date - 20/07/2021
The module performs detection for timed water heater
"""

# Import python packages
import logging
import numpy as np
from copy import deepcopy
from datetime import datetime

# Import packages from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.hld_checks import hld_checks
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.twh_maths_utils import vertical_filter
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.twh_maths_utils import horizontal_filter
from python3.disaggregation.aer.waterheater.functions.timed_japan.timed_wh_config import init_twh_config
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.box_fitting import box_fitting
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.twh_detection import twh_detection
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.timed_wh_utils import get_2d_matrix
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.perform_post_processing import post_processing
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.get_consistency_arr import get_consistency_arr
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.check_heating_instance import check_heating_instance
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.get_gaussian_band_data import get_gaussian_band_data


def timed_waterheater_detection(in_data, debug, sampling_rate, logger_base):
    """
    This function is used to perfom timed wh detection
    Parameters:
        in_data             (np.ndarray)        : Input data
        debug               (dict)              : Algorithm steps output
        sampling_rate       (int)               : Sampling rate of the data
        logger_base         (dict)              : Dictionary containing the logger object and logging dict

    Returns:
        twh_estimation      (dict)              : Final Timed Wh signal
        debug               (dict)              : Algorithm steps output
    """

    start = datetime.now()

    # Initialise logger for this module

    logger_local = logger_base.get('logger').getChild('timed_waterheater_detection')
    logger_pass = {'logger': logger_local, 'logging_dict': logger_base.get('logging_dict')}
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Take the deepcopy of input data

    input_data = deepcopy(in_data)
    debug['input_data'] = deepcopy(in_data)

    # Initialise the wh config

    wh_config = init_twh_config(sampling_rate, debug)

    # Convert the 21 column data to 2D matrix

    data_matrix, row_idx, col_idx, ytick_labels, day_points = get_2d_matrix(input_data, sampling_rate, debug)

    # Store the 2D matrix to 21 column data row mapping in debug object for future reference

    debug['row_idx'] = row_idx
    debug['col_idx'] = col_idx
    debug['day_points'] = day_points
    debug['ytick_labels'] = ytick_labels
    debug['rows'] = data_matrix.shape[0]
    debug['cols'] = data_matrix.shape[1]
    debug['factor'] = Cgbdisagg.SEC_IN_HOUR/sampling_rate

    # ----------------------------------------- STEP-1 Clean the data -------------------------------------------------#

    # Perform Vertical and Horizontal filtering

    v_percentile = wh_config['vertical_percentile']
    h_percentile = wh_config['horizontal_percentile']
    v_window_size = wh_config['vertical_window_size']
    h_window_size = wh_config['horizontal_window_size']

    t1 = datetime.now()

    vertical_filtered_data = vertical_filter(data_matrix, window_size=v_window_size, percentile=v_percentile)
    horizontal_filtered_data = horizontal_filter(data_matrix, window_size=h_window_size, percentile=h_percentile, sampling_rate=sampling_rate)

    t2 = datetime.now()
    logger.info('Cleaning the data took | %.3f s ', get_time_diff(t1, t2))

    # Get the cleaned data

    cleaned_data = vertical_filtered_data - horizontal_filtered_data
    cleaned_data[cleaned_data < 0] = 0

    # ----------------------------------------- STEP-2 Get the consistent time intervals -----------------------------#

    t1 = datetime.now()

    concentration_arr_bool, concentration_arr = get_consistency_arr(cleaned_data, wh_config, debug)
    debug['concentration_arr'] = concentration_arr
    debug['concentration_arr_bool'] = concentration_arr_bool

    t2 = datetime.now()
    logger.info('Get consistency array took | %.3f s ', get_time_diff(t1, t2))

    # Initialise necessary debug variables

    debug['num_runs'] = 0
    debug['timed_hld'] = 0
    debug['timed_confidence'] = 0
    debug['timed_wh_amplitude'] = 0

    # Check if the detection module has to be run or not

    run_detection_bool = False
    if debug.get('disagg_mode') == 'mtd' and debug.get('hsm_in').get('timed_hld') == [0]:
        run_detection_bool = False
    elif np.sum(concentration_arr_bool):
        run_detection_bool = True

    logger.info('Status of sufficient time bands found to run the Japan timed wh | {}'.format(run_detection_bool))

    if run_detection_bool:

        # Make the non consistent time interval data to 0 for detection

        twh_data_matrix = data_matrix.copy()
        aoi_filtered_data = cleaned_data.copy()
        twh_data_matrix[:, ~concentration_arr_bool] = 0
        aoi_filtered_data[:, ~concentration_arr_bool] = 0

        # -------------------------------------- STEP-3 Check for heating instance -----------------------------------#

        t1 = datetime.now()
        aoi_filtered_data, twh_data_matrix = check_heating_instance(aoi_filtered_data, twh_data_matrix, debug,
                                                                    wh_config, logger_pass)

        t2 = datetime.now()
        logger.info('Check heating instance took | %.3f s ', get_time_diff(t1, t2))

        # -------------------------------------- STEP-4 Get Gaussian band data ---------------------------------------#

        t1 = datetime.now()
        scored_data_matrix, overall_chunk_data = get_gaussian_band_data(aoi_filtered_data, concentration_arr_bool,
                                                                        debug, wh_config, logger_pass)

        t2 = datetime.now()
        logger.info('Get gaussian band data took | %.3f s ', get_time_diff(t1, t2))

        # save all the step wise output matrices in debug

        debug['cleaned_data'] = deepcopy(cleaned_data)
        debug['aoi_data'] = deepcopy(aoi_filtered_data)
        debug['twh_data_matrix'] = deepcopy(twh_data_matrix)
        debug['original_data_matrix'] = deepcopy(data_matrix)
        debug['scored_data_matrix'] = deepcopy(scored_data_matrix)

        # -------------------------------------- STEP-5 Perform Box Fitting ------------------------------------------#

        if len(overall_chunk_data):

            t1 = datetime.now()

            debug = box_fitting(debug, wh_config, logger_pass)

            t2 = datetime.now()
            logger.info('Box fitting took | %.3f s ', get_time_diff(t1, t2))

            # -------------------------------------- STEP-6 Timed WH detection --------------------------------------#

            t1 = datetime.now()

            debug = twh_detection(debug, wh_config, logger_pass)

            t2 = datetime.now()
            logger.info('TWH detection took | %.3f s ', get_time_diff(t1, t2))

        # -------------------------------------- STEP-7 Post Processing ----------------------------------------------#

        t1 = datetime.now()
        if debug['timed_hld']:
            final_twh_matrix, debug = post_processing(debug, wh_config, logger_pass)
            hld, twh_confidence, debug = hld_checks(debug, wh_config)

            if not hld:
                debug['num_runs'] = 0
                debug['timed_hld'] = 0
                debug['timed_confidence'] = twh_confidence
                debug['timed_wh_amplitude'] = 0
                final_twh_matrix[:, :] = 0

            logger.info('Timed WH confidence after HLD checks is {} | '.format(debug['timed_confidence']))

        t2 = datetime.now()
        logger.info('Post processing time took | %.3f s ', get_time_diff(t1, t2))

        # -------------------------------------- STEP-8 Final User check ----------------------------------------------#

    logger.info('Timed Water heater detection status is {} | '.format(debug['timed_hld']))

    # Fill the timed wh estimation

    twh_estimation = deepcopy(in_data)
    twh_estimation[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    if debug['timed_hld'] == 1:
        twh_estimation[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = final_twh_matrix[row_idx, col_idx].flatten()

    end = datetime.now()
    logger.info('TWH module took | %.3f s ', get_time_diff(start, end))
    return twh_estimation, debug
