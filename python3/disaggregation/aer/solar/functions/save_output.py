"""
Author - Paras Tehria
Date - 12/11/19
This module saves solar detection files locally
"""

# Import python packages

import os
import copy
import shutil
import logging
import numpy as np
import pandas as pd

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.path_constants import PathConstants
from python3.utils.maths_utils.maths_utils import forward_fill
from python3.utils.maths_utils.maths_utils import create_pivot_table
from python3.utils.visualization.generate_heatmap_so_detection import dump_heatmap_raw


def save_csv(uuid, input_data, user_dir, so_config, csv_flag, prob_arr):
    """
    Function to save local csv files

        Parameters:
            input_data          (np.ndarray)      : Algorithm intermediate steps output
            uuid                (str)             : User id
            user_dir            (str)             : saving directory of a user
            so_config            (dict)            : module config parameters
            csv_flag             (bool)            : bool signifying whether to save user csv file
            prob_arr            (list)            : instance probability list

        Return:
            y_signal_raw        (np.ndarray)      : 2-d raw consumption data where each row is a day
            y_signal_sun        (np.ndarray)      : 2-d sunlight presence data where each row is a day
            prob_arr            (list)            : Instance probability array
            conf_arr            (list)            : Confidence array of all the runs
    """
    raw_csv = uuid + '_y_signal_raw.csv'
    sun_csv = uuid + '_y_signal_sun.csv'
    prob_csv = uuid + '_det_prob.csv'
    conf_csv = uuid + '_det_conf.csv'

    # Save raw data to csv file

    y_signal_raw, _, _ = create_pivot_table(data=input_data, index=Cgbdisagg.INPUT_DAY_IDX,
                                            columns=Cgbdisagg.INPUT_HOD_IDX, values=Cgbdisagg.INPUT_CONSUMPTION_IDX)

    # Replacing nan values with the value at the same on the previous day (ffill followed by bfill)
    y_signal_raw = forward_fill(y_signal_raw)

    # bfill
    y_signal_raw = np.flipud(forward_fill(np.flipud(y_signal_raw)))

    # Generating solar presence pivot table

    sun_index = so_config.get('prep_solar_data').get('sun_index')
    y_signal_sun, _, _= create_pivot_table(data=input_data, index=Cgbdisagg.INPUT_DAY_IDX,
                                           columns=Cgbdisagg.INPUT_HOD_IDX, values=sun_index)

    # Replacing nan values with the value at the same on the previous day (ffill followed by bfill)

    y_signal_sun = forward_fill(y_signal_sun)
    y_signal_sun = np.flipud(forward_fill(np.flipud(y_signal_sun)))

    num_cols = y_signal_raw.shape[1]

    padded_row = np.full(shape=(10, num_cols), fill_value=0)

    # Add the padded row to the start and end of the data matrix
    y_signal_raw = np.r_[y_signal_raw, padded_row]
    y_signal_sun = np.r_[y_signal_sun, padded_row]

    # Convert numpy array to data frames
    # Save the data frames to the target location
    pd.DataFrame(data=y_signal_raw).to_csv(user_dir + '/' + raw_csv, mode='a', header=None, index=None)
    pd.DataFrame(data=y_signal_sun).to_csv(user_dir + '/' + sun_csv, mode='a', header=None, index=None)
    pd.DataFrame(data=prob_arr).to_csv(user_dir + '/' + prob_csv, mode='a', header=None, index=None)

    max_instances = so_config.get('max_instances')

    prob_arr = pd.read_csv(user_dir + '/' + prob_csv, header=None).values.flatten()

    # removing disconnection indices
    prob_arr_after_disconn = prob_arr[prob_arr != -2]

    if len(prob_arr_after_disconn) > 0:
        conf_arr = [np.round(np.mean(prob_arr_after_disconn[-max_instances:]), 2)]
    else:
        conf_arr = [0.00]

    conf_df = pd.DataFrame(data=conf_arr)
    conf_df.to_csv(user_dir + '/' + conf_csv, mode='a', header=None, index=None)
    conf_arr = pd.read_csv(user_dir + '/' + conf_csv, header=None).values.flatten()

    return y_signal_raw, y_signal_sun, prob_arr, conf_arr


def save_output(global_config, disagg_input_object, so_config, input_data, confidence, probability_solar, logger_base, start_date, end_date, kind):
    """
    This functions saves heatmap if flag is true

    Parameters:
        global_config        (dict)       : global config dictionary
        disagg_input_object (dict)        : disagg input object
        so_config            (dict)       : solar config dictionary
        input_data          (np.ndarray)  : input data matrix
        confidence          (float)       : LGBM confidence_output
        probability_solar   (list)        : instance level probability
        logger_base         (dict)        : Logger object to write logs
        start_date          (float)       : Start date of solar generation
        end_date            (float)       : End date of solar generation
        kind                (string)      : Solar panel present throughout or installation/removal

    Returns:
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('save_output')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Extract user information for saving files

    uuid = global_config.get('uuid')

    # Check which all output need to be saved

    csv_flag = 'solar' in global_config.get('dump_csv')
    plots_flag = 'solar' in global_config.get('generate_plots')

    # Make directory with uuid as the name for dumping s0_output

    user_dir = PathConstants.LOG_DIR + PathConstants.MODULE_OUTPUT_FILES_DIR['solar'] + uuid

    # Check if any of the solar output needs to be saved

    debug_bool = csv_flag | plots_flag

    # If any of the so_output is required to be saved, create user directory

    if disagg_input_object.get('config').get('disagg_mode') == 'historical' and os.path.exists(user_dir):
        shutil.rmtree(user_dir)

    if (not os.path.exists(user_dir)) and debug_bool:
        os.makedirs(user_dir)

    logger.info('CSV write - %s, Plots generate - %s | ', str(csv_flag), str(plots_flag))

    # Check if solar heatmaps are needed to dump

    logger.info('Solar plot dump: {} | '.format(plots_flag))

    if (csv_flag or plots_flag) & (start_date is not None):
        logger.info('Dumping solar detection heatmaps and csvs | ')

        y_signal_raw, y_signal_sun, prob_arr, conf_arr = save_csv(uuid, copy.deepcopy(input_data), user_dir, so_config,
                                                                  csv_flag, probability_solar)
        logger.info('CSV files saved locally | ')

        # Get starting and ending day for solar presence
        days = np.unique(input_data[:,Cgbdisagg.INPUT_DAY_IDX])
        days = np.where((start_date <= days) & (days <= end_date))[0]

        if (plots_flag) and (len(days) > 0):
            start = days[0]
            end = days[-1]
            dump_heatmap_raw(y_signal_raw, y_signal_sun, so_config, prob_arr, conf_arr, user_dir, confidence, start, end, kind)
            logger.info('heatmaps saved locally | ')

    return
