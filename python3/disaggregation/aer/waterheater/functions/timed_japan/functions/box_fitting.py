"""
Author - Sahana M
Date - 07/06/2021
Perform box fitting
"""

# Import python packages
import logging
import numpy as np
from copy import deepcopy

# Import packages from within the project
from python3.disaggregation.aer.waterheater.functions.timed_japan.timed_wh_config import windows
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.get_box_data import get_box_data
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.convert_box_data_to_2d import box_data_to_2d
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.merge_sequences import merge_sequences
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.dynamic_box_cleaning import dynamic_box_cleaning


def get_window_boxes(filtered_data, wh_config):
    """
    For each window size the raw data is fitted with boxes
    Parameters:
        filtered_data           (np.ndarray)   : 2D matrix containing the raw data corresponding to each chunk
        wh_config               (dict)        : WH configurations dictionary
    Returns:
        box_fitting_data        (dict)        : For each window size has the start row, start time, end row, end time, amplitude info of all the boxes
    """

    # Initialise the necessary data

    box_fitting_data = {}
    factor = wh_config.get('factor')
    window_size_array = windows[factor]
    box_min_amp = wh_config.get('box_min_amp')
    auc_min_amp_criteria = wh_config.get('auc_min_amp_criteria')
    auc_max_amp_criteria = wh_config.get('auc_max_amp_criteria')
    repetition_arr = np.full_like(filtered_data, fill_value=0)

    for window_size in window_size_array:

        # Get box fitted data

        box_info, box_val = get_box_data(filtered_data, window_size)

        # Check for Area Under Curve minimum criteria

        low_consumption_boxes = box_val * window_size < (auc_min_amp_criteria / factor) * window_size
        box_val[low_consumption_boxes] = 0

        # Check for Area Under Curve maximum criteria

        high_consumption_boxes = box_val * window_size > auc_max_amp_criteria
        box_val[high_consumption_boxes] = 0

        box_val_2d = box_val.reshape(filtered_data.shape[0], filtered_data.shape[1])

        # Find out the number of repetitions

        repetition_arr = repetition_arr + (box_val_2d > 0) * 1

        # Minimum amplitude check

        box_val_2d[box_val_2d <= box_min_amp/factor] = 0

        # Store the data in a dictionary
        box_fitting_data[window_size] = box_info

    return box_fitting_data


def box_fitting(debug, wh_config, logger_base):
    """
    Box fitting function is used to fit boxes on the raw data for each identified chunk sequence
    Parameters:
        debug                   (dict)              : Contains algorithm output
        wh_config               (dict)              : WH configurations dictionary
        logger_base             (logger)            : Logger passed

    Returns:
        debug                   (dict)              : Contains algorithm output
    """

    # Initialise logger for this module

    logger_local = logger_base.get('logger').getChild('box_fitting')
    logger_pass = {'logger': logger_local, 'logging_dict': logger_base.get('logging_dict')}
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Extract the necessary data

    rows = debug.get('rows')
    days = wh_config.get('days')
    cols = wh_config.get('cols')
    factor = wh_config.get('factor')
    cleaned_data = debug.get('cleaned_data')
    amp_bar_twh = wh_config.get('amp_bar_twh')
    sliding_window = wh_config.get('sliding_window')
    data_matrix = deepcopy(debug.get('twh_data_matrix'))
    scored_data_matrix = debug.get('scored_data_matrix')
    overall_chunk_data = debug.get('overall_chunk_data')
    original_data_matrix = debug.get('original_data_matrix')

    # Make sure to cap the high amplitude data
    amplitude_cap = int(amp_bar_twh / factor)
    data_matrix[data_matrix > amplitude_cap] = amplitude_cap

    # ----------------------------------- Step - 1 Copy raw data info ------------------------------------------------#

    # Copy all the raw data for each chunk into filtered_data array

    filtered_data = np.full_like(data_matrix, fill_value=0.0)
    start_idx_arr = np.arange(0, cleaned_data.shape[0] - sliding_window, sliding_window).astype(int)
    for i in start_idx_arr:
        start_idx = int(i)
        end_idx = int(min((start_idx + days), cleaned_data.shape[0]))
        temp_data = deepcopy(data_matrix[start_idx:end_idx, :])
        score_data_idx = int(np.where(start_idx_arr == i)[0])
        scored_data_matrix_bool = scored_data_matrix[score_data_idx, :] > 0
        temp_data[:, ~scored_data_matrix_bool] = 0
        filtered_data[start_idx:end_idx, :] = temp_data

    # ----------------------------------- Step - 2 Perform box fitting -----------------------------------------------#

    box_fitting_data = get_window_boxes(filtered_data, wh_config)

    logger.info('Box fitting for each window size completed | ')

    # ----------------------------------- Step - 3 Dynamic selection of best boxes -----------------------------------#

    cleaned_data_matrix, overall_selected_boxes = dynamic_box_cleaning(box_fitting_data, filtered_data, overall_chunk_data,
                                                                       wh_config)

    logger.info('Number of best fit boxes found | {}'.format(len(overall_selected_boxes)))

    # ----------------------------------- Step - 4 Convert to 2D matrix ----------------------------------------------#

    cleaned_data_matrix = box_data_to_2d(overall_selected_boxes, rows, cols)

    masked_cleaned_data = deepcopy(filtered_data)
    masked_cleaned_data[~(cleaned_data_matrix > 0)] = 0

    # ----------------------------------- Step - 5 Merge chunk instances ---------------------------------------------#

    overall_chunk_data = merge_sequences(overall_chunk_data, cleaned_data_matrix, wh_config, logger_pass)

    debug['start_idx_arr'] = start_idx_arr
    debug['box_fit_matrix'] = cleaned_data_matrix
    debug['overall_chunk_data'] = overall_chunk_data
    debug['scored_data_matrix'] = scored_data_matrix
    debug['masked_cleaned_data'] = masked_cleaned_data
    debug['original_data_matrix'] = original_data_matrix

    return debug
