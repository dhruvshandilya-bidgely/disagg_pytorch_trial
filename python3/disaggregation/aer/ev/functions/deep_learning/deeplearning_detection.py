"""
Author - Sahana M
Date - 14-Nov-2023
Module performing Deep learning predictions
"""

import logging
from datetime import datetime
import numpy as np
import pandas as pd
from copy import deepcopy

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff
from python3.disaggregation.aer.ev.functions.deep_learning.init_dl_config import init_ev_params
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import get_hod_matrix
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import get_day_data_2d
from python3.disaggregation.aer.ev.functions.deep_learning.get_ev_l2_detection import get_ev_l2_detection
from python3.disaggregation.aer.ev.functions.deep_learning.get_ev_l1_detection import get_ev_l1_user_detection
from python3.disaggregation.aer.ev.functions.deep_learning.get_potential_l2_boxes import get_potential_l2_boxes
from python3.disaggregation.aer.ev.functions.deep_learning.get_potential_l1_boxes import get_potential_l1_boxes
from python3.disaggregation.aer.ev.functions.deep_learning.get_partition_predictions import get_partition_predictions


def initialise_required_variables(data_matrices, row_idx, col_idx, ml_debug, ev_config):

    """
    Function to initialise the required variables for Deep learning debugging
    Parameters:
        data_matrices                   (np.ndarray)            : 2D matrix of all the columns in the input data
        row_idx                         (np.ndarray)            : Row indexes mapping of 1D input data to 2D
        col_idx                         (np.ndarray)            : Column indexes mapping of 1D input data to 2D
        ml_debug                        (Dict)                  : Machine learning debugger object
        ev_config                       (Dict)                  : EV configurations
    Returns:
        dl_debug                        (Dict)                  : Deep learning debugger object
    """

    # Initialise the required variables

    raw_data = data_matrices[Cgbdisagg.INPUT_CONSUMPTION_IDX]
    heat_pot_data = data_matrices[Cgbdisagg.INPUT_HEATING_POTENTIAL_IDX]
    cool_pot_data = data_matrices[Cgbdisagg.INPUT_COOLING_POTENTIAL_IDX]
    temp_data = data_matrices[Cgbdisagg.INPUT_TEMPERATURE_IDX]
    s_label_data = data_matrices[Cgbdisagg.INPUT_S_LABEL_IDX]

    # Reshape the data as required

    rows, cols = raw_data.shape
    heat_pot_data = np.asarray(pd.DataFrame(heat_pot_data.flatten()).ffill()).reshape(rows, cols)
    cool_pot_data = np.asarray(pd.DataFrame(cool_pot_data.flatten()).ffill()).reshape(rows, cols)
    temp_data = np.asarray(pd.DataFrame(temp_data.flatten()).ffill()).reshape(rows, cols)
    s_label_data = np.asarray(pd.DataFrame(s_label_data.flatten()).ffill()).reshape(rows, cols)
    hvac_removed_1d = deepcopy(raw_data[row_idx, col_idx])

    # Extract required info

    factor = int(ml_debug.get('factor'))
    sampling_rate = ev_config.get('sampling_rate')

    # Get the hour of the day column

    hod_matrix = get_hod_matrix(raw_data, factor)

    # Perform partition operation variables

    initial_cols = raw_data.shape[1]
    total_days = len(raw_data)
    total_partitions = int(np.ceil(len(raw_data) / 14))

    # Initialise the configurations variable

    config = init_ev_params()

    dl_debug = {
        'config': config,
        'factor': factor,
        'row_idx': row_idx,
        'col_idx': col_idx,
        'ev_config': ev_config,
        'temperature': temp_data,
        'total_days': total_days,
        'hod_matrix': hod_matrix,
        'heat_pot': heat_pot_data,
        'cool_pot': cool_pot_data,
        's_label_data': s_label_data,
        'initial_cols': initial_cols,
        'raw_data': deepcopy(raw_data),
        'sampling_rate': sampling_rate,
        'input_raw_data': data_matrices,
        'hvac_removed': deepcopy(raw_data),
        'hvac_removed_1d': hvac_removed_1d,
        'total_partitions': total_partitions,
    }

    dl_debug['config']['region'] = ev_config.get('region')

    return dl_debug


def deeplearning_detection(data_matrices, ml_debug, ev_config, row_idx, col_idx, logger_base):
    """
    Function to run the deep learning model for EV Detection
    Parameters:
        data_matrices               (np.ndarray)            : Data matrix
        ml_debug                       (Dict)               : Debug dictionary
        ev_config                   (Dict)                  : EV configurations dictionary
        row_idx                     (np.ndarray)            : Boolean row index array
        col_idx                     (np.ndarray)            : Boolean column index array
        logger_base                 (logger)                : Logger passed
    Returns:
        dl_debug             (Dict)                  : Debug partition dictionary
    """

    # Initialise the logger
    logger_local = logger_base.get('logger').getChild('deep_learning_exp')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))
    logger_pass = {'logger': logger_local, 'logging_dict': logger_base.get('logging_dict')}

    # Initialise the required variables

    dl_debug = initialise_required_variables(data_matrices, row_idx, col_idx, ml_debug, ev_config)

    # ---------------------------------- PARTITION PREDICTIONS - L2 ---------------------------------------------------

    t_start = datetime.now()
    predictions_bool, prediction_confidences = get_partition_predictions(dl_debug, ml_debug, logger_pass)
    t_end = datetime.now()

    logger.info('DL L2 : Partitions predicted | %s ', np.sum(predictions_bool))
    logger.info('DL L2 : Partitions predictions complete | Time taken %s ', get_time_diff(t_start, t_end))

    predictions_bool = np.asarray(predictions_bool)
    prediction_confidences = np.asarray(prediction_confidences)

    dl_debug['predictions_bool'] = predictions_bool
    dl_debug['final_partition_predictions'] = predictions_bool
    dl_debug['prediction_confidences'] = prediction_confidences

    # ---------------------------------- POTENTIAL EV BOXES EXTRACTION - L2 -------------------------------------------

    t_start = datetime.now()
    final_boxes_detected, dl_debug = get_potential_l2_boxes(dl_debug, logger_pass)
    dl_debug['potential_l2_boxes'] = deepcopy(final_boxes_detected)
    t_end = datetime.now()

    logger.info('DL L2 : Potential EV boxes extracted | Time taken %s ', get_time_diff(t_start, t_end))

    # ---------------------------------------------- EV L2 Detection at User level ------------------------------------

    t_start = datetime.now()
    final_boxes_detected, dl_debug = get_ev_l2_detection(final_boxes_detected, dl_debug, ml_debug, logger_pass)
    t_end = datetime.now()

    logger.info('DL L2 : User level EV detection complete | Time taken %s ', get_time_diff(t_start, t_end))

    # ---------------------------------------------- EV L1 Detection --------------------------------------------------

    if dl_debug['combined_hld'] == 0 and ml_debug.get('disagg_mode') != 'mtd' and len(ml_debug.get('l1')):

        # Get the 2D matrix

        data_matrices_l1, row_idx, col_idx = get_day_data_2d(ml_debug.get('hvac_removed_data_l1'), ev_config)

        raw_data_l1 = deepcopy(data_matrices_l1[Cgbdisagg.INPUT_CONSUMPTION_IDX])
        hvac_removed_1d_l1 = deepcopy(raw_data_l1[row_idx, col_idx])
        dl_debug['raw_data_l1'] = raw_data_l1

        # ---------------------------------- PARTITION PREDICTIONS - L1 -----------------------------------------------

        t_start = datetime.now()
        predictions_bool_l1, prediction_confidences_l1 = get_partition_predictions(dl_debug, ml_debug, logger_pass, ctype='L1')
        t_end = datetime.now()

        logger.info('DL L1 : Partitions predicted | %s ', np.sum(predictions_bool_l1))
        logger.info('DL L1 : Partitions predictions complete | Time taken %s ', get_time_diff(t_start, t_end))

        # Assign the required variables

        predictions_bool_l1 = np.asarray(predictions_bool_l1)
        prediction_confidences_l1 = np.asarray(prediction_confidences_l1)
        dl_debug['hvac_not_removed'] = deepcopy(raw_data_l1)
        dl_debug['final_partition_predictions'] = predictions_bool
        dl_debug['final_partition_predictions_l1'] = predictions_bool_l1
        dl_debug['predictions_bool_l1'] = predictions_bool_l1
        dl_debug['prediction_confidences_l1'] = prediction_confidences_l1
        dl_debug['hvac_removed_1d_l1'] = hvac_removed_1d_l1

        # ---------------------------------- POTENTIAL EV BOXES EXTRACTION - L1 ----------------------------------------

        t_start = datetime.now()
        final_boxes_detected, dl_debug = get_potential_l1_boxes(raw_data_l1, dl_debug, logger_pass)
        dl_debug['potential_l1_boxes'] = deepcopy(final_boxes_detected)
        t_end = datetime.now()

        logger.info('DL L1 : Potential EV boxes extracted | Time taken %s ', get_time_diff(t_start, t_end))

        # ---------------------------------------------- EV L1 Detection at User level --------------------------------

        dl_debug['low_density_l1'] = 'NA'
        dl_debug['user_detection_l1'] = 0
        dl_debug['user_confidence_l1'] = np.round(0, 2)

        # Get the final L1 detection

        t_start = datetime.now()
        final_boxes_detected, dl_debug = get_ev_l1_user_detection(final_boxes_detected, dl_debug, ml_debug, logger_pass)
        t_end = datetime.now()

        logger.info('DL L1 : User level EV detection complete | Time taken %s ', get_time_diff(t_start, t_end))

    dl_debug['final_boxes_detected'] = final_boxes_detected

    return dl_debug
