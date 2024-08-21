"""
Author - Mayank Sharan
Date - 7/12/19
This file holds functions to dump plots, csv and the debug object as specified by the config
"""

# Import python packages

import os
import pytz
import pickle
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.path_constants import PathConstants

# Import pyplot from matplotlib after setting Agg backend that should work on almost all machines and servers
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def run_vacation_debug(debug, vacation_config, global_config):

    """
    Function to dump plots, csv and debug object as per configuration

    Parameters:
        debug               (dict)              : Contains all variables needed for debugging
        vacation_config     (dict)              : Dictionary containing all needed configuration variables
        global_config       (dict)              : Dictionary containing global configuration variables
    """

    # Extract flags from the global config

    dump_csv = 'va' in global_config.get('dump_csv')
    dump_debug = 'va' in global_config.get('dump_debug')
    dump_plots = 'va' in global_config.get('generate_plots')

    # Initialize config variables as required

    user_info_config = vacation_config.get('user_info')

    uuid = user_info_config.get('uuid')
    timezone = user_info_config.get('tz')
    sampling_rate = user_info_config.get('sampling_rate')

    # If none of the dump plots, dump debug or dump csv flags are true return

    if not (dump_plots or dump_csv or dump_debug):
        return

    # Initialize the root directory for dumping stuff as per the priority flag and initialize config

    if global_config.get('priority'):
        root_dir = PathConstants.LOG_DIR_PRIORITY
    else:
        root_dir = PathConstants.LOG_DIR

    res_dir = root_dir + 'vac_res/' + user_info_config.get('uuid') + '/'

    # If the directory does not exist create it

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    if dump_csv:

        # Dump vacation periods

        vac_periods = debug.get('vacation_periods')

        vac_periods_df = pd.DataFrame(data=vac_periods)
        vac_periods_df.to_csv(res_dir + 'vacation_periods_' + uuid + '.csv', index=None, header=None)

    # If dump debug flag is true save the debug dictionary

    if dump_debug:

        # Open a file and dump the debug dictionary as a pickle

        pickle_file = open(res_dir + 'debug_' + uuid + '.pb', 'wb')
        pickle.dump(debug, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    # Initialize and create if needed plots dir

    plots_dir = res_dir + 'day_plots/'

    if not os.path.exists(plots_dir) and dump_plots:
        os.makedirs(plots_dir)

    # Extract basic variables from debug dictionary

    day_ts = debug.get('data').get('day_ts')

    day_data = debug.get('data').get('day_data')
    day_data_processed = debug.get('data').get('day_data_processed')

    day_valid_mask_cons = debug.get('data').get('day_valid_mask_cons')

    power = debug.get('features').get('day_wise_power')
    baseload = debug.get('features').get('day_wise_baseload')

    vac_label = debug.get('labeling').get('label')
    vac_confidence = debug.get('labeling').get('confidence')

    probable_vac_bool = debug.get('labeling').get('probable_vac_bool')
    probable_threshold = debug.get('labeling').get('probable_threshold')

    power_chk_thr = debug.get('labeling').get('power_chk_thr')
    loop_break_idx = debug.get('labeling').get('loop_break_idx')
    loop_break_power = debug.get('labeling').get('loop_break_power')
    power_check_passed_bool = debug.get('labeling').get('power_check_passed_bool')

    sliding_mean_thr = debug.get('labeling').get('sliding_mean_thr')
    sliding_power_mean_all = debug.get('labeling').get('sliding_power_mean_all')

    std_dev_all = debug.get('labeling').get('std_dev_all')
    std_dev_thr = debug.get('labeling').get('std_dev_thr')

    max_3_dev_all = debug.get('labeling').get('max_3_dev_all')
    perc_arr_all = debug.get('labeling').get('perc_arr_all')

    sliding_power = debug.get('labeling').get('sliding_power')

    # Compute values as needed

    frac_pd = sampling_rate / Cgbdisagg.SEC_IN_HOUR

    # Initialize variables needed to dump plots and / or dump csv

    num_days = day_data.shape[0]

    day_str_arr = np.zeros(shape=(num_days,), dtype=object)
    max_3_dev_str = np.zeros(shape=(num_days,), dtype=object)
    sliding_power_str = np.zeros(shape=(num_days,), dtype=object)
    loop_break_power_str = np.zeros(shape=(num_days,), dtype=object)

    tz = pytz.timezone(timezone)
    x_vector = np.arange(0, Cgbdisagg.HRS_IN_DAY, frac_pd)

    label_1 = 'input_data'
    label_2 = 'processed_data'

    # Initialize first column index for day_ts array

    day_ts_first_col = 0

    # Compute day string for each day and if flag is on dump plots

    for day_idx in range(num_days):

        # Get the day string and populate it into the day string array

        day_string = datetime.fromtimestamp(day_ts[day_idx, day_ts_first_col], tz).strftime('%Y-%m-%d')
        day_str_arr[day_idx] = day_string

        # Populate other strings as needed

        max_3_dev_str[day_idx] = ' '.join(max_3_dev_all[day_idx, :].astype(str))
        sliding_power_str[day_idx] = ' '.join(sliding_power[day_idx, :].astype(str))
        loop_break_power_str[day_idx] = ' '.join(loop_break_power[day_idx, :].astype(str))

        if dump_plots:

            # Initialize figure

            plt.figure(figsize=(15, 10))

            # Plot the line plots

            plt.plot(x_vector, day_data[day_idx, :], color='blue', label=label_1, linestyle='--')
            plt.plot(x_vector, day_data_processed[day_idx, :], color='red', label=label_2, linestyle='-.')

            # Create the title string with information from different debug variables

            title_string = day_string + '  label : ' + str(vac_label[day_idx]) + \
                '  confidence : ' + str(vac_confidence[day_idx]) + \
                '\nbaseload : ' + str(baseload[day_idx]) + \
                '  power : ' + str(power[day_idx]) + \
                '  prob_thr : ' + str(probable_threshold[day_idx]) + \
                '  prob_label : ' + str(probable_vac_bool[day_idx]) + \
                '\npower_chk_thr : ' + str(power_chk_thr[day_idx]) + \
                '  loop_break_idx : ' + str(loop_break_idx[day_idx]) + \
                '  loop_break_power : ' + loop_break_power_str[day_idx] + \
                '  loop_passed : ' + str(power_check_passed_bool[day_idx]) + \
                '\npower_mean : ' + str(sliding_power_mean_all[day_idx]) + \
                '  mean_thr : ' + str(sliding_mean_thr[day_idx]) + \
                '  std_dev : ' + str(std_dev_all[day_idx]) + \
                '  std_dev_thr : ' + str(std_dev_thr[day_idx]) + \
                '  max_3_dev : ' + str(max_3_dev_str[day_idx]) + \
                '\nsliding power arr : ' + sliding_power_str[day_idx]

            # Label the X and Y axes and put up the title string

            plt.xlabel('Hour of the day')
            plt.ylabel('Consumption (Wh)')
            plt.title(title_string)

            # Add grid and legend to the plot

            plt.grid()
            plt.legend()

            # Save the plot

            plt.savefig(plots_dir + day_string + '_' + uuid + '.png')
            plt.close()

    if dump_csv:

        # Compute the number of hours in a day that are masked as timed devices

        day_masked_hrs = np.sum(day_valid_mask_cons, axis=1) * frac_pd

        # Initialize the day info array

        day_info = np.full(shape=(num_days, 19), fill_value=np.nan, dtype=object)

        # Populate the day info array with all relevant variables

        day_info[:, 0] = day_str_arr

        day_info[:, 1] = vac_label
        day_info[:, 2] = vac_confidence

        day_info[:, 3] = day_masked_hrs

        day_info[:, 4] = baseload
        day_info[:, 5] = power

        day_info[:, 6] = probable_threshold
        day_info[:, 7] = probable_vac_bool

        day_info[:, 8] = power_chk_thr
        day_info[:, 9] = loop_break_idx
        day_info[:, 10] = loop_break_power_str
        day_info[:, 11] = power_check_passed_bool

        day_info[:, 12] = sliding_power_mean_all
        day_info[:, 13] = sliding_mean_thr

        day_info[:, 14] = std_dev_all
        day_info[:, 15] = std_dev_thr

        day_info[:, 16] = max_3_dev_str
        day_info[:, 17] = perc_arr_all

        day_info[:, 18] = sliding_power_str

        # Save the populated data in the csv

        day_info_df = pd.DataFrame(data=day_info, columns=['Date (Y-m-d)', 'Vacation Label', 'Vac Confidence',
                                                           'Num Hrs Masked', 'Baseload (Wh)', 'Power (Wh)',
                                                           'Probable Thr (Wh)', 'Probable Vacation',
                                                           'Power Chk Thr (Wh)', 'Loop break idx',
                                                           'Loop break power (Wh)', 'Power Chk Passed',
                                                           'Power mean (Wh)', 'Power Mean Thr (Wh)',
                                                           'Std dev (Wh)', 'Std dev Thr (Wh)',
                                                           'Max dev (Wh)', 'Max 3 dev perc', 'Sliding Power'])

        day_info_df.to_csv(res_dir + 'all_info_' + uuid + '.csv', index=None)
