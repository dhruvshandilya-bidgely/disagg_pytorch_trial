"""
Author - Sahana M
Date - 15-Nov-2023
Module performing partition predictions
"""

# import python packages
import datetime
import gc
import logging
import numpy as np
from copy import deepcopy

# tensorflow GPU disabling
# GPU disabling
# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
# from tensorflow import convert_to_tensor, float32
import torch
torch.cuda.is_available = lambda: False
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import packages from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import rolling_function


def padding(data, shape=[28, 28]):
    """
    Function to perform padding
    Parameters:
        data                    (np.ndarray)        : Data for padding
        shape                   (list)              : Shape
    Returns:
        data                    (np.ndarray)        : Data for padding
    """

    # Extract the required variables
    input_rows = data.shape[0]
    input_cols = data.shape[1]
    padding_rows = int((shape[0] - input_rows) / 2)
    padding_cols = int((shape[1] - input_cols) / 2)

    # add padding rows both up and down
    zero_rows = np.zeros((padding_rows, input_cols))
    zero_cols = np.zeros((shape[0], padding_cols))

    # add the zero rows and columns to the numpy array
    data = np.r_[zero_rows, data, zero_rows]
    data = np.c_[zero_cols, data, zero_cols]

    return data


def downsample(initial_load):
    """
    Function to downsample the given data
    Parameters:
        initial_load                (np.ndarray)            : Initial load
    Returns:
        new_load                    (np.ndarray)            : Downsampling to 1 hour
    """

    row, col = initial_load.shape
    s_r = col//Cgbdisagg.HRS_IN_DAY
    new_load = np.zeros([row, Cgbdisagg.HRS_IN_DAY])
    for i in range(row):
        for j in range(0, col, s_r):
            temp = sum(initial_load[i, j:j+s_r])
            new_load[i, j//s_r] = temp
    return new_load


def remove_standalone(load):
    """
    Function to remove standalone points
    Parameters:
        load                    (np.ndarray)            : Input data
    Returns:
        res_load                (np.ndarray)            : Residual load data
    """

    # Extract the required variables

    initial_load = deepcopy(load)
    new_init = np.ndarray.flatten(initial_load)
    temp_day_bool = new_init > 0
    start_end = np.diff(np.r_[0, temp_day_bool.astype(int), 0])
    start = np.where(start_end == 1)[0]
    end = np.where(start_end == -1)[0]
    for i in range(len(start)):
        if abs(end[i] - start[i]) == 1:
            new_init[start[i]: end[i]] = 0

    res_load = np.reshape(new_init, (len(initial_load), len(initial_load[0])))
    return res_load


def baseload_removal(initial_load, window_hours=Cgbdisagg.HRS_IN_DAY, ctype='L2'):
    """
    Function to perform baseload removal
    Parameters:
        initial_load                    (np.ndarray)            : Initial load
        window_hours                    (int)                   : Window hours
        ctype                            (string)                : Charger types
    Returns:
        new_load                        (np.ndarray)            : New load
    """
    # Extract the required variables

    row = initial_load.shape[0]
    col = initial_load.shape[1]
    sampling_rate = Cgbdisagg.SEC_IN_HOUR / (initial_load.shape[1] / Cgbdisagg.HRS_IN_DAY)

    if ctype == 'L1':
        window_hours = Cgbdisagg.HRS_IN_DAY * 2

    temp = initial_load.flatten()
    window_size = window_hours * Cgbdisagg.SEC_IN_HOUR / sampling_rate
    window_size = np.floor(window_size / 2) * 2 + 1

    # Perform min max rolling

    base_load = rolling_function(temp, window_size, 'min')
    base_load = rolling_function(base_load, window_size, 'max')

    base_load = base_load.reshape(row, col)

    new_load = np.subtract(initial_load, base_load)
    return new_load


def pre_proc(initial_load, ctype='L2'):
    """
    Function to perform preprocessing
    Parameters:
        initial_load                (np.ndarray)            : Initial load
        ctype                       (str)                   : Charger type
    Returns:
        final_load                  (np.ndarray)            : Final load
    """

    # Perform base-load removal, removing standalone points, down-sampling and padding

    new_rows = 28
    new_cols = 28
    new_load = baseload_removal(initial_load, Cgbdisagg.HRS_IN_DAY, ctype)
    final_load = remove_standalone(new_load)
    downsampled_load = downsample(final_load)
    final_load = padding(downsampled_load, shape=[new_rows, new_cols])

    return final_load


def get_partition_predictions(dl_debug, ml_debug, logger_base, ctype='L2'):
    """
    Function to get the predictions for each partition
    Parameters:
        dl_debug                        (Dict)          : Deep learning debug dictionary
        ml_debug                        (Dict)          : Machine learning debug dictionary
        ctype                           (string)        : Charger type
    Returns:
        predictions_bool                (np.ndarray)    : Predictions boolean array
        prediction_confidences          (np.ndarray)    : Prediction confidences array
    """

    # Initialise the logger
    logger_local = logger_base.get('logger').getChild('get_potential_l2_boxes')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Initialise the required variables

    predictions_bool = []
    prediction_confidences = []
    raw_data = dl_debug.get('raw_data')
    total_days = dl_debug.get('total_days')
    initial_cols = dl_debug.get('initial_cols')
    prt_size = dl_debug.get('config').get('prt_size')
    total_partitions = dl_debug.get('total_partitions')
    prt_conf_thr = dl_debug.get('config').get('prt_conf_thr')
    prt_conf_thr_l1 = dl_debug.get('config').get('prt_conf_thr_l1')

    # Charger wise thresholds

    conf_threshold = prt_conf_thr
    if ctype == 'L1':
        conf_threshold = prt_conf_thr_l1
        raw_data = dl_debug.get('raw_data_l1')

    if ctype == 'L2':
        model = ml_debug.get('models').get('ev_hld').get('l2_cnn')
        model = model.eval()
        model = model.float()
    else:
        model = ml_debug.get('models').get('ev_hld').get('l1_cnn')
        model = model.eval()
        model = model.float()

    # For each partition perform preprocessing and predictions
    logger.info("DL %s : Total partitions to predict | %s", ctype, total_partitions)
    print('Model loading done.')

    for i in range(total_partitions):

        # Perform padding if required

        padding = False
        padding_rows = []
        start_day = i * prt_size
        end_day = start_day + prt_size
        if end_day >= total_days:
            padding = True
            padding_days = end_day - total_days
            padding_rows = np.zeros((padding_days, initial_cols))
        curr_partition = raw_data[start_day: end_day, :]
        if padding:
            curr_partition = np.r_[curr_partition, padding_rows]

        # Perform preprocessing

        curr_partition = pre_proc(curr_partition, ctype)
        curr_partition = curr_partition.reshape(1, prt_size*2, prt_size*2, 1)
        tf_input = torch.from_numpy(curr_partition)

        # Perform model predictions

        t_start = datetime.datetime.now()
        with torch.no_grad():
            out_data = model(tf_input.float())
        prediction = np.asarray(out_data)
        t_end = datetime.datetime.now()
        logger.info('DL %s : Individual partition prediction | %s', ctype, get_time_diff(t_start, t_end))

        # Assign the prediction values
        if prediction[0] >= conf_threshold:
            prediction_bool = 1
        else:
            prediction_bool = 0
        predictions_bool.append(prediction_bool)
        prediction_confidences.append(prediction[0])

    t_start = datetime.datetime.now()
    _ = model(tf_input.float())
    _ = gc.collect()
    t_end = datetime.datetime.now()
    logger.info('DL %s : Garbage clearing time | %s', ctype, get_time_diff(t_start, t_end))

    return predictions_bool, prediction_confidences
