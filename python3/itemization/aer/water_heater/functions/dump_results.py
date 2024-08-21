"""
Author - Sahana M
Date - 23/4/2021
This file holds functions to dump plots, csv and the debug object as specified by the config
"""

# Import python packages
import os
import pickle
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
from datetime import datetime, date

# Import pyplot from matplotlib after setting Agg backend that should work on almost all machines and servers
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.path_constants import PathConstants


def dump_results(debug, seasonal_wh_config, global_config):
    """
    Function to dump plots, csv and debug object as per configuration
    Args:
        debug               (dict)          : Contains all variables needed for debugging
        seasonal_wh_config  (dict)          : Contains all seasonal wh configurations
        global_config       (dict)          : Contains all global configurations
    """

    # Extract flags from the global config

    dump_csv = 'wh' in global_config.get('dump_csv')
    dump_debug = 'wh' in global_config.get('dump_debug')
    dump_plots = 'wh' in global_config.get('generate_plots')

    # Extract required variables

    uuid = seasonal_wh_config['user_info']['uuid']
    pilot = seasonal_wh_config['user_info']['pilot_id']
    sampling_rate = seasonal_wh_config['user_info']['sampling_rate']
    pipeline_mode = global_config.get('disagg_mode')

    # If none of the dump plots, dump debug or dump csv flags are true return

    if not (dump_plots or dump_csv or dump_debug):
        return

    # Initialize the root directory for dumping stuff as per the priority flag and initialize config

    if global_config.get('priority'):
        root_dir = PathConstants.LOG_DIR_PRIORITY + PathConstants.MODULE_OUTPUT_FILES_DIR['wh'] + uuid + '/swh/'
    else:
        root_dir = PathConstants.LOG_DIR + PathConstants.MODULE_OUTPUT_FILES_DIR['wh'] + uuid + '/swh/'

    res_dir = root_dir

    # If the directory does not exist create it

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    if dump_debug:

        # Open a file and dump the debug dictionary as a pickle
        last_ts = debug.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX]
        pickle_file = open(res_dir + 'swh_debug_' + uuid + '_' + pipeline_mode + '_' + str(last_ts) + '.pb', 'wb')
        pickle.dump(debug, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    # Initialize and create if needed plots dir

    plots_dir = res_dir

    if not os.path.exists(plots_dir) and dump_plots:
        os.makedirs(plots_dir)

    if dump_plots:

        # Initialise all the required variables

        input_data = deepcopy(debug['input_data'])
        swh_cleaned = deepcopy(debug['swh_cleaned_data'])

        # Change the input_data hour segmented column to hour

        swh_cleaned[:, Cgbdisagg.INPUT_HOD_IDX] = deepcopy(input_data[:, Cgbdisagg.INPUT_HOD_IDX])

        # get the hours in a day

        num_pd_in_day = int(Cgbdisagg.SEC_IN_DAY / sampling_rate)

        pd_mult = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

        # Prepare day timestamp matrix and get size of all matrices

        day_ts, row_idx = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)
        day_ts = np.tile(day_ts, reps=(num_pd_in_day, 1)).transpose()

        # Initialize all 2d matrices with default value of nan except the boolean ones

        input_data_matrix = np.full(shape=day_ts.shape, fill_value=np.nan)
        swh_cleaned_matrix = np.full(shape=day_ts.shape, fill_value=np.nan)

        # Compute hour of day based indices to use

        col_idx = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] - input_data[:, Cgbdisagg.INPUT_DAY_IDX]
        col_idx = col_idx / Cgbdisagg.SEC_IN_HOUR
        col_idx = (pd_mult * (col_idx - col_idx.astype(int) + input_data[:, Cgbdisagg.INPUT_HOD_IDX])).astype(int)

        # Create day wise 2d arrays for each variable

        input_data_matrix[row_idx, col_idx] = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
        swh_cleaned_matrix[row_idx, col_idx] = swh_cleaned[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

        max_cap = np.nanpercentile(input_data_matrix, q=99.9)
        input_data_matrix[input_data_matrix > max_cap] = max_cap

        max_cap = np.nanpercentile(swh_cleaned_matrix, q=99.9)
        swh_cleaned_matrix[swh_cleaned_matrix > max_cap] = max_cap

        max0 = np.nanmax(input_data_matrix)
        max1 = np.nanmax(swh_cleaned_matrix)

        # convert the numpy arrays to pandas dataframe for column naming

        input_data_df = pd.DataFrame(input_data_matrix)
        swd_cleaned_df = pd.DataFrame(swh_cleaned_matrix)

        # Initialise x ticks column names and assign the dataframes with the column names (time stamps)

        frac_pd = Cgbdisagg.HRS_IN_DAY / input_data_df.shape[1]

        hour_tick = np.arange(0, Cgbdisagg.HRS_IN_DAY, frac_pd)
        frac_tick = (hour_tick - np.floor(hour_tick)) * 0.6
        day_points = (np.floor(hour_tick) + frac_tick).astype(int)

        input_data_df.columns = day_points
        swd_cleaned_df.columns = day_points

        # Initialise y ticks native

        yticks = day_ts[:, 0]
        ytick_labels = []

        for j in range(0, len(yticks)):
            dt = datetime.fromtimestamp(yticks[j])
            dv = datetime.timetuple(dt)
            month = date(int(dv[0]), int(dv[1]), int(dv[2])).strftime('%d-%b-%y')
            ytick_labels.append(month)

        # Copy y labels

        input_data_df.index = ytick_labels
        swd_cleaned_df.index = ytick_labels

        fig_heatmap, axn = plt.subplots(1, (4 + debug['total_detections'] + 1), sharey=True)
        fig_heatmap.set_size_inches(30, 20)
        fig_heatmap.suptitle('uuid : ' + uuid
                             + ', Pilot : ' + str(pilot)
                             + ', Sampling Rate: ' + str(sampling_rate) + 's'
                             + 'Correlation : ' + str(debug['swh_correlation'])
                             + ' Max correlation : ' + str(np.nanmax(debug['swh_correlation'])),
                             fontsize=16)

        # Plot the heatmap

        axn.flat[0].set_ylabel(ytick_labels)
        axn.flat[4].set_ylabel(ytick_labels)
        sns.heatmap(input_data_df, cmap='jet', ax=axn.flat[0], cbar=True, xticklabels=8, yticklabels=30, vmin=0,
                    vmax=max0)
        sns.heatmap(swd_cleaned_df, cmap='jet', ax=axn.flat[1], cbar=True, xticklabels=8, yticklabels=30, vmin=0,
                    vmax=max1)

        # Plot WH potential and Feels like heatmap

        wh_pot = debug.get('wh_potential')
        num_wh_pot_days = len(wh_pot[wh_pot > 0])
        num_days = len(wh_pot)
        wh_pot = pd.DataFrame(wh_pot)
        sns.heatmap(wh_pot, cmap='Blues', ax=axn.flat[2])
        temperature = debug.get('fl')
        temperature = pd.DataFrame(temperature)
        sns.heatmap(temperature, cmap='jet', ax=axn.flat[3])

        # For every band detected plot the heatmap

        for i in range(debug['total_detections']):
            data_matrix = debug['swh_run' + str(i) + '_estimation']
            temp_max = np.nanmax(data_matrix)
            data_df = pd.DataFrame(data_matrix)
            data_df.columns = day_points
            data_df.index = ytick_labels
            sns.heatmap(data_df, cmap='jet', ax=axn.flat[4 + i], cbar=True, xticklabels=8, yticklabels=30, vmin=0,
                        vmax=temp_max)

        # Intersection of times zones and wh days

        intersection_data_matrix = debug['final_estimation']
        final_max = np.nanmax(intersection_data_matrix)
        inter_data_df = pd.DataFrame(intersection_data_matrix)
        inter_data_df.columns = day_points
        inter_data_df.index = ytick_labels
        sns.heatmap(inter_data_df, cmap='jet', ax=axn.flat[(4 + debug['total_detections'])], cbar=True, xticklabels=8,
                    yticklabels=30, vmin=0,
                    vmax=final_max)

        # Label for the plots

        axn.flat[0].set_title('Original Data')
        axn.flat[1].set_title('Cleaned Data')
        axn.flat[2].set_title('WH Potential' + '\nWH pot days : ' + str(int(num_wh_pot_days)) + '/' + str(int(num_days)))
        axn.flat[3].set_title('Feels Like Temperature')
        for i in range(debug['total_detections']):
            axn.flat[i + 4].set_title(
                '  \nband_corr : ' + str(debug[str('swh_run') + str(i) + '_band_corr'])
                + '\nWrong_days : ' + str(debug[str('swh_run') + str(i) + '_wrong_days'])
                + '\nTb prob : ' + str(debug[str('swh_run') + str(i) + '_tb_prob'])
                + '\nWtr consis : ' + str(debug[str('swh_run' + str(i)) + '_winter_consistency'])
                + '\nmax_median : ' + str(debug[str('swh_run') + str(i) + '_max_median'])
                + '\nEnergy diff : ' + str(debug[str('swh_run') + str(i) + '_energy_diff'])
                + '\nE range : ' + str(debug[str('swh_run') + str(i) + '_e_range'])
                + '\nConsumption : ' + str(debug[str('swh_run' + str(i)) + '_consumption'])
                + '\nScore : ' + str(debug[str('swh_run') + str(i) + '_score']))
        axn.flat[4 + debug['total_detections']].set_title('Final estimation'
                                                          + '\nBands detected : ' + str(debug['total_detections'])
                                                          + '\nEnergy range : ' + str(debug['final_e_range'])
                                                          + '\nTotal consumption : ' + str(debug['final_consumption']))
        # Align ticks and their orientation

        axn.flat[0].tick_params(axis='y', labelrotation=0)
        axn.flat[1].tick_params(axis='y', labelrotation=0)
        for i in range(debug['total_detections']):
            axn.flat[i + 4].tick_params(axis='y', labelrotation=0)
        axn.flat[4 + debug['total_detections']].tick_params(axis='y', labelrotation=0)

        # Save the plot

        save_dir = plots_dir + str(uuid) + '_' + str(pipeline_mode) + '.png'
        plt.savefig(save_dir)
        plt.close()
