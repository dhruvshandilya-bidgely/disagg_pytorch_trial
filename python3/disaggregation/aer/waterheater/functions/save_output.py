"""
Author - Nikhil Singh Chauhan
Date - 02/11/18
This module gives the functionality to :
    1. Dump the water heater estimate in a CSV file
    2. Dump the debug dictionary as a pickle file
    3. Generate the heatmap of input and output data
"""

# Import python packages

import os
import copy
import pickle
import logging
import numpy as np
import pandas as pd

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.file_utils import file_zip
from python3.utils.file_utils import file_unzip
from python3.config.path_constants import PathConstants
from python3.disaggregation.aer.waterheater.functions.get_day_data import get_day_data, get_day_data_twh
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.twh_debug_heatmaps import band_plots
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.twh_debug_heatmaps import estimation_plots
from python3.utils.visualization.waterheater_heatmaps import wh_heatmaps


def save_output(global_config, debug, out_bill_cycles, timezone, logger_base):
    """
    This functions checks the configuration and saves all the wh_output
    asked for.

    Parameters:
        global_config       (dict)      : Parameters dictionary
        debug               (dict)      : Debug dictionary with values recorded throughout the code run
        out_bill_cycles     (list)      : Bill cycles for which output is to be given
        timezone            (int)       : Timezone in hours w.r.t. GMT
        logger_base         (dict)      : Logger object to write logs

    Returns:
        Saves the relevant files without returning anything
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('save_output')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Extract user information for saving files

    uuid = global_config.get('uuid')
    pilot = global_config.get('pilot_id')
    sampling_rate = global_config.get('sampling_rate')

    # Check which all output need to be saved

    padding_zero = True
    dump_twh_csv = False

    csv_flag = 'wh' in global_config.get('dump_csv')
    debug_flag = 'wh' in global_config.get('dump_debug')
    plots_flag = 'wh' in global_config.get('generate_plots')

    # Make directory with uuid as the name for dumping wh_output

    user_dir = PathConstants.LOG_DIR + PathConstants.MODULE_OUTPUT_FILES_DIR['wh'] + uuid

    # Check if any of the 3 water heater output needs to be saved

    debug_bool = csv_flag | debug_flag | plots_flag

    # If any of the wh_output is required to be saved, create user directory

    if (not os.path.exists(user_dir)) and debug_bool:
        os.makedirs(user_dir)

    user_dir += '/'

    logger.info('CSV write - %s, Debug write - %s, Plots generate - %s | ', str(csv_flag), str(debug_flag),
                str(plots_flag))

    # Check the config if CSV files need to be saved

    if csv_flag and dump_twh_csv:

        # Save 2D data csv files

        save_twh_csv(debug, uuid, sampling_rate, out_bill_cycles, timezone, user_dir, padding_zero, logger_pass)

        logger.info('Timed WH CSV files saved locally | ')

    elif csv_flag and (debug['timed_hld'] != 1):
        # Save lap info

        save_laps(debug, user_dir)

        # Save 2D data csv files

        save_csv(debug, uuid, sampling_rate, out_bill_cycles, timezone, user_dir, padding_zero, logger_pass)

        logger.info('CSV files saved locally | ')
    else:
        logger.info('CSV files not saved locally | ')

    # Check the config if debug object need to be saved

    if debug_flag:
        # Dumpy debug object

        save_debug(debug, uuid, user_dir)

        logger.info('Debug dictionary saved locally | ')
    else:
        logger.info('Debug dictionary not saved locally | ')

    # Check the config if plots need to be saved

    if plots_flag:

        if debug['timed_hld'] == 1:
            # If timed water heater

            logger.info('Saving plots for timed water heater | ')

            wh_heatmaps(debug['input_data'], debug['timed_wh_signal'], uuid, sampling_rate, user_dir, pilot=pilot,
                        tag='timed', conf=np.max(debug['timed_debug']['timed_confidence']), num_runs=debug['timed_debug']['num_runs'])

        elif debug['thermostat_hld'] == 1:
            # If thermostat water heater

            logger.info('Saving plots for non-timed water heater | ')

            wh_heatmaps(debug['input_data'], debug['final_wh_signal'], uuid, sampling_rate, user_dir, pilot=pilot,
                        tag='thermostat', conf=debug["thermostat_hld_prob"])

        else:
            # If no water heater found

            wh_output = copy.deepcopy(debug['input_data'])
            wh_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

            wh_heatmaps(debug['input_data'], wh_output, uuid, sampling_rate, user_dir, pilot=pilot,
                        tag='none', conf=0)

            logger.info('Plots/Heatmaps saved locally | ')

    # For local japan timed wh debugging

    timed_debugging_plots = False

    if plots_flag and timed_debugging_plots:

        band_plots(debug, uuid, sampling_rate, user_dir, logger_pass, pilot=pilot)

        estimation_plots(debug, uuid, sampling_rate, user_dir, pilot=pilot)

        logger.info('Timed band plots/Heatmaps saved locally | ')

    return


def save_csv(debug, uuid, sampling_rate, out_bill_cycles, timezone, save_path, padding_zero, logger_base):
    """
    Parameters:
        debug               (dict)      : Algorithm intermediate steps output
        uuid                (str)       : User id
        sampling_rate       (int)       : Sampling rate of the user
        out_bill_cycles     (list)      : List of the bill cycles for which output is to be saved
        timezone            (int)       : Timezone in hours w.r.t. GMT
        save_path           (str)       : Path to save output files
        padding_zero        (bool)      : If the bill cycle boundaries to be of zeros or max
        logger_base         (logger)    : Logger object
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('save_csv')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve the relevant files from the debug object

    raw = debug['input_data']
    thin = debug['final_thin_output']
    fat = debug['final_fat_output']
    residual = debug['residual']

    # Fill NaNs with zero

    raw[np.isnan(raw)] = 0
    thin[np.isnan(thin)] = 0
    fat[np.isnan(fat)] = 0
    residual[np.isnan(residual)] = 0

    # Cap the negative values with zero

    raw = np.fmax(raw, 0)
    thin = np.fmax(thin, 0)
    fat = np.fmax(fat, 0)
    residual = np.fmax(residual, 0)

    # Update current working directory to uuid folder

    current_working_dir = os.getcwd()
    original_pipeline_path = copy.deepcopy(current_working_dir)

    logger.info('Current Working directory | {}'.format(current_working_dir))

    # noinspection PyBroadException
    try:
        # Check if the current working directory changes

        os.chdir(current_working_dir + '/' + save_path)
    except ValueError:
        # If the current working directory didn't change, get the default one

        current_working_dir = '/'
        original_pipeline_path = copy.deepcopy(current_working_dir)
        os.chdir(current_working_dir + '/' + save_path)

    logger.info('The new current working directory | {}'.format(current_working_dir + '/' + save_path))

    # Unzip the files (if present in the target location)

    file_unzip(logger)

    # Save raw data to csv file

    raw_data = copy.deepcopy(raw)
    pd.DataFrame(raw_data).to_csv(uuid + '_input_matrix.csv', index=False)

    # Convert the raw data and water heater output to 2d-matrices where each row represents single day

    daily_data_output = get_day_data(raw, thin, fat, residual, timezone, sampling_rate, padding_zero)

    # Retrieve all the corresponding matrices from the output list

    month_ts, day_ts, epoch_ts, input_data, thin_data, fat_data, wh_data, residual_data = daily_data_output

    # Replace NaN with zero for all the output data tables

    day_ts[np.isnan(day_ts)] = 0
    wh_data[np.isnan(wh_data)] = 0
    month_ts[np.isnan(month_ts)] = 0
    epoch_ts[np.isnan(epoch_ts)] = 0
    fat_data[np.isnan(fat_data)] = 0
    thin_data[np.isnan(thin_data)] = 0
    input_data[np.isnan(input_data)] = 0
    residual_data[np.isnan(residual_data)] = 0

    # Replace Inf / -Inf with zero for all the output data tables

    day_ts[np.isinf(np.abs(day_ts))] = 0
    wh_data[np.isinf(np.abs(wh_data))] = 0
    month_ts[np.isinf(np.abs(month_ts))] = 0
    epoch_ts[np.isinf(np.abs(epoch_ts))] = 0
    fat_data[np.isinf(np.abs(fat_data))] = 0
    thin_data[np.isinf(np.abs(thin_data))] = 0
    input_data[np.isinf(np.abs(input_data))] = 0
    residual_data[np.isinf(np.abs(residual_data))] = 0

    # Declare the file names for all the saved outputs

    wh_csv = uuid + '_wh.csv'
    fat_csv = uuid + '_fat.csv'
    thin_csv = uuid + '_thin.csv'
    day_ts_csv = uuid + '_day_ts.csv'
    input_data_csv = uuid + '_input.csv'
    month_ts_csv = uuid + '_month_ts.csv'
    epoch_ts_csv = uuid + '_epoch_ts.csv'
    residual_csv = uuid + '_residual.csv'

    # Check which days data is to be retained

    days_to_include = np.in1d(np.nanmin(month_ts, axis=1), out_bill_cycles)

    # Subset the data based on the required days found above

    wh_copy = wh_data[days_to_include, :].copy()
    fat_copy = fat_data[days_to_include, :].copy()
    day_ts_copy = day_ts[days_to_include, :].copy()
    thin_copy = thin_data[days_to_include, :].copy()
    month_ts_copy = month_ts[days_to_include, :].copy()
    epoch_ts_copy = epoch_ts[days_to_include, :].copy()
    day_data_copy = input_data[days_to_include, :].copy()
    residual_copy = residual_data[days_to_include, :].copy()

    # Get the number of columns in the 2D matrix

    num_cols = fat_copy.shape[1]

    if padding_zero == True:
        padded_row = np.full(shape=(1, num_cols), fill_value=0)
    else:
        padded_row = np.full(shape=(1, num_cols), fill_value=np.max(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))

    # Add the padded row to the start and end of the data matrix

    wh_copy = np.r_[padded_row, wh_copy, padded_row]
    fat_copy = np.r_[padded_row, fat_copy, padded_row]
    thin_copy = np.r_[padded_row, thin_copy, padded_row]
    day_data_copy = np.r_[padded_row, day_data_copy, padded_row]
    residual_copy = np.r_[padded_row, residual_copy, padded_row]

    month_ts_copy = np.r_[np.reshape(month_ts[0, :], newshape=(1, num_cols)), month_ts_copy,
                          np.reshape(month_ts[-1, :], newshape=(1, num_cols))]
    day_ts_copy = np.r_[np.reshape(day_ts[0, :], newshape=(1, num_cols)), day_ts_copy,
                        np.reshape(day_ts[-1, :], newshape=(1, num_cols))]
    epoch_ts_copy = np.r_[np.reshape(epoch_ts[0, :], newshape=(1, num_cols)), epoch_ts_copy,
                          np.reshape(epoch_ts[-1, :], newshape=(1, num_cols))]

    # Convert numpy array to data frames

    wh_df = pd.DataFrame(data=wh_copy)
    fat_df = pd.DataFrame(data=fat_copy)
    thin_df = pd.DataFrame(data=thin_copy)
    day_ts_df = pd.DataFrame(data=day_ts_copy)
    month_ts_df = pd.DataFrame(data=month_ts_copy)
    day_data_df = pd.DataFrame(data=day_data_copy)
    epoch_ts_df = pd.DataFrame(data=epoch_ts_copy)
    residual_df = pd.DataFrame(data=residual_copy)

    # Save the data frames to the target location

    wh_df.to_csv(wh_csv, mode='a', header=None, index=None)
    fat_df.to_csv(fat_csv, mode='a', header=None, index=None)
    thin_df.to_csv(thin_csv, mode='a', header=None, index=None)
    day_ts_df.to_csv(day_ts_csv, mode='a', header=None, index=None)
    month_ts_df.to_csv(month_ts_csv, mode='a', header=None, index=None)
    epoch_ts_df.to_csv(epoch_ts_csv, mode='a', header=None, index=None)
    residual_df.to_csv(residual_csv, mode='a', header=None, index=None)
    day_data_df.to_csv(input_data_csv, mode='a', header=None, index=None)

    # Zipping files to a folder for memory reasons

    file_zip(logger)

    # Update the current working directory back to the pipeline

    os.chdir(original_pipeline_path)

    logger.info('The restored current working directory | {}'.format(original_pipeline_path))

    return


def save_twh_csv(debug, uuid, sampling_rate, out_bill_cycles, timezone, save_path, padding_zero, logger_base):
    """
    This function is used to save the Timed water heater csv
    Parameters:
        debug               (dict)      : Algorithm intermediate steps output
        uuid                (str)       : User id
        sampling_rate       (int)       : Sampling rate of the user
        out_bill_cycles     (list)      : List of the bill cycles for which output is to be saved
        timezone            (int)       : Timezone in hours w.r.t. GMT
        save_path           (str)       : Path to save output files
        padding_zero        (bool)      : If the bill cycle boundaries to be of zeros or max
        logger_base         (logger)    : Logger object
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('save_csv')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve the relevant files from the debug object

    raw = debug['input_data']
    temp = debug['input_data']
    wh = debug['timed_wh_signal']
    residual = debug['timed_debug']['twh_residual']

    # Fill NaNs with zero

    wh[np.isnan(wh)] = 0
    raw[np.isnan(raw)] = 0
    temp[np.isnan(temp)] = 0
    residual[np.isnan(residual)] = 0

    # Cap the negative values with zero

    wh = np.fmax(wh, 0)
    raw = np.fmax(raw, 0)
    temp = np.fmax(temp, 0)
    residual = np.fmax(residual, 0)

    # Update current working directory to uuid folder

    current_working_dir = os.getcwd()
    original_pipeline_path = copy.deepcopy(current_working_dir)

    logger.info('Current Working directory | {}'.format(current_working_dir))

    # noinspection PyBroadException
    try:
        # Check if the current working directory changes

        os.chdir(current_working_dir + '/' + save_path)
    except ValueError:
        # If the current working directory didn't change, get the default one

        current_working_dir = '/'
        original_pipeline_path = copy.deepcopy(current_working_dir)
        os.chdir(current_working_dir + '/' + save_path)

    logger.info('The new current working directory | {}'.format(current_working_dir + '/' + save_path))

    # Unzip the files (if present in the target location)

    file_unzip(logger)

    # Save raw data to csv file

    raw_data = copy.deepcopy(raw)
    pd.DataFrame(raw_data).to_csv(uuid + '_input_matrix.csv', index=False)

    # Convert the raw data and water heater output to 2d-matrices where each row represents single day

    daily_data_output = get_day_data_twh(raw, temp, wh, residual, timezone, sampling_rate, padding_zero)

    # Retrieve all the corresponding matrices from the output list

    month_ts, day_ts, epoch_ts, input_data, temp_data, wh_data, residual_data = daily_data_output

    # Replace NaN with zero for all the output data tables

    day_ts[np.isnan(day_ts)] = 0
    month_ts[np.isnan(month_ts)] = 0
    epoch_ts[np.isnan(epoch_ts)] = 0
    input_data[np.isnan(input_data)] = 0
    wh_data[np.isnan(wh_data)] = 0
    temp_data[np.isnan(temp_data)] = 0
    residual_data[np.isnan(residual_data)] = 0

    # Replace Inf / -Inf with zero for all the output data tables

    day_ts[np.isinf(np.abs(day_ts))] = 0
    month_ts[np.isinf(np.abs(month_ts))] = 0
    epoch_ts[np.isinf(np.abs(epoch_ts))] = 0
    input_data[np.isinf(np.abs(input_data))] = 0
    wh_data[np.isinf(np.abs(wh_data))] = 0
    temp_data[np.isinf(np.abs(temp_data))] = 0
    residual_data[np.isinf(np.abs(residual_data))] = 0

    # Declare the file names for all the saved outputs

    day_ts_csv = uuid + '_day_ts.csv'
    input_data_csv = uuid + '_input.csv'
    month_ts_csv = uuid + '_month_ts.csv'
    epoch_ts_csv = uuid + '_epoch_ts.csv'
    wh_data_csv = uuid + '_wh.csv'
    temp_data_csv = uuid + '_temperature.csv'
    residual_data_csv = uuid + '_residual.csv'

    # Check which days data is to be retained

    days_to_include = np.in1d(np.nanmin(month_ts, axis=1), out_bill_cycles)

    # Subset the data based on the required days found above

    wh_copy = wh_data[days_to_include, :].copy()
    temp_copy = temp_data[days_to_include, :].copy()
    day_ts_copy = day_ts[days_to_include, :].copy()
    month_ts_copy = month_ts[days_to_include, :].copy()
    epoch_ts_copy = epoch_ts[days_to_include, :].copy()
    day_data_copy = input_data[days_to_include, :].copy()
    residual_copy = residual_data[days_to_include, :].copy()

    # Get the number of columns in the 2D matrix

    num_cols = wh_copy.shape[1]

    if padding_zero == True:
        padded_row = np.full(shape=(1, num_cols), fill_value=0)
    else:
        padded_row = np.full(shape=(1, num_cols), fill_value=np.max(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))

    # Add the padded row to the start and end of the data matrix

    wh_copy = np.r_[padded_row, wh_copy, padded_row]
    day_data_copy = np.r_[padded_row, day_data_copy, padded_row]
    residual_copy = np.r_[padded_row, residual_copy, padded_row]
    temp_copy = np.r_[padded_row, temp_copy, padded_row]

    month_ts_copy = np.r_[np.reshape(month_ts[0, :], newshape=(1, num_cols)), month_ts_copy,
                          np.reshape(month_ts[-1, :], newshape=(1, num_cols))]
    day_ts_copy = np.r_[np.reshape(day_ts[0, :], newshape=(1, num_cols)), day_ts_copy,
                        np.reshape(day_ts[-1, :], newshape=(1, num_cols))]
    epoch_ts_copy = np.r_[np.reshape(epoch_ts[0, :], newshape=(1, num_cols)), epoch_ts_copy,
                          np.reshape(epoch_ts[-1, :], newshape=(1, num_cols))]

    # Convert numpy array to data frames

    wh_df = pd.DataFrame(data=wh_copy)
    day_ts_df = pd.DataFrame(data=day_ts_copy)
    month_ts_df = pd.DataFrame(data=month_ts_copy)
    day_data_df = pd.DataFrame(data=day_data_copy)
    epoch_ts_df = pd.DataFrame(data=epoch_ts_copy)
    residual_df = pd.DataFrame(data=residual_copy)
    temp_df = pd.DataFrame(data=temp_copy)

    # Save the data frames to the target location

    wh_df.to_csv(wh_data_csv, mode='a', header=None, index=None)
    day_ts_df.to_csv(day_ts_csv, mode='a', header=None, index=None)
    month_ts_df.to_csv(month_ts_csv, mode='a', header=None, index=None)
    epoch_ts_df.to_csv(epoch_ts_csv, mode='a', header=None, index=None)
    day_data_df.to_csv(input_data_csv, mode='a', header=None, index=None)
    residual_df.to_csv(residual_data_csv, mode='a', header=None, index=None)
    temp_df.to_csv(temp_data_csv, mode='a', header=None, index=None)

    # Zipping files to a folder for memory reasons

    file_zip(logger)

    # Update the current working directory back to the pipeline

    os.chdir(original_pipeline_path)

    logger.info('The restored current working directory | {}'.format(original_pipeline_path))

    return


def save_debug(debug_dict, uuid, save_path):
    """
    This function saved the debug dictionary in the pickle file format

    Parameters:
        debug_dict      (dict)          : Debug dictionary with values recorded throughout the code run
        uuid            (string)        : User ID
        save_path       (string)        : path where file are to be saved
    """

    last_ts = debug_dict.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX]

    file_handle = open(save_path + uuid + '_' + debug_dict.get('disagg_mode') +  '_' + str(last_ts) + '_debug.pkl', 'wb')

    # Save the debug dict in the file

    pickle.dump(debug_dict, file_handle)

    # Close the file

    file_handle.close()

    return


def save_laps(debug, save_path):
    """
    Parameters:
        debug           (dict)      : Algorithm intermediate steps output
        save_path       (str)       : Path to save lap files
    """

    lap_path = save_path + 'laps/'

    # Create the directory for saving lap info

    if (not os.path.exists(lap_path)):
        os.makedirs(lap_path)

    # Retrieve the seasonal information from debug object

    season_features = debug['season_features']

    # Define the list of columns for LAP data

    lap_columns = ['bill_cycle_ts', 'day_ts', 'start_ts', 'end_ts', 'duration']

    # Iterate over each season

    for season in season_features.keys():
        # Retrieve the laps info for the season

        laps = season_features[season]['laps']

        # If valid number of laps found, save the laps file

        if laps.shape[0] > 0:

            # Convert numpy array to data-frame

            lap_info_df = pd.DataFrame(laps)

            # Add column names to the data-frame

            lap_info_df.columns = lap_columns

            # Save the seasonal lap info to given location

            lap_info_df.to_csv(lap_path + season + '_laps.csv', index=False)

    return
