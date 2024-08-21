"""
Author - Nikhil Singh Chauhan
Date - 15-May-2020
This module gives the functionality to :
    1. Dump the water heater estimate in a CSV file
    2. Dump the debug dictionary as a pickle file
    3. Generate the heatmap of input and output data
"""

# Import python packages

import os
import pickle
import logging
import numpy as np
import pandas as pd
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.file_utils import file_zip
from python3.utils.file_utils import file_unzip
from python3.config.path_constants import PathConstants
from python3.disaggregation.aer.ev.functions.get_day_data import get_day_data

from python3.utils.visualization.ev_heatmaps import ev_heatmaps


def save_ev_output(global_config, debug, out_bill_cycles, ev_config, logger_base):
    """
    This functions checks the configuration and saves all the EV output
    asked for.

    Parameters:
        global_config       (dict)       : Parameters dictionary
        debug               (dict)      : Debug dictionary with values recorded throughout the code run
        out_bill_cycles     (list)      : Bill cycles for which output is to be given
        ev_config            (dict)      : EV config params
        logger_base         (dict)      : Logger object to write logs

    Returns:
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

    region = ev_config.get('region')
    timezone = ev_config.get('timezone_hours')
    disagg_mode = ev_config.get('disagg_mode')

    # Check which all output need to be saved

    padding_zero = True

    csv_flag = 'ev' in global_config.get('dump_csv')
    debug_flag = 'ev' in global_config.get('dump_debug')
    plots_flag = 'ev' in global_config.get('generate_plots')

    # If the plots flag is True then csv_flag should be True

    if plots_flag:
        csv_flag = True

    # Make directory with uuid as the name for dumping EV output

    user_dir = PathConstants.LOG_DIR + PathConstants.MODULE_OUTPUT_FILES_DIR['ev'] + uuid

    # Check if any of the 3 types of output needs to be saved

    debug_bool = csv_flag | debug_flag | plots_flag

    # If any of the ev output is required to be saved, create user directory

    if (not os.path.exists(user_dir)) and debug_bool:
        os.makedirs(user_dir)

    user_dir += '/'

    logger.info('CSV write - %s, Debug write - %s, Plots generate - %s | ', str(csv_flag), str(debug_flag),
                str(plots_flag))

    # Check the config if CSV files need to be saved

    if csv_flag:

        # Save 2D data csv files and detection features

        save_csv(debug, uuid, sampling_rate, out_bill_cycles, timezone, user_dir, padding_zero, disagg_mode,
                 logger_pass)

        logger.info('CSV files and features saved locally | ')
    else:
        logger.info('CSV files and features not saved locally | ')

    # Check the config if debug object need to be saved

    if debug_flag:
        # Dumpy debug object

        save_debug(debug, uuid, user_dir)

        logger.info('Debug dictionary saved locally | ')
    else:
        logger.info('Debug dictionary not saved locally | ')

    # Check the config if plots need to be saved

    if plots_flag:

        required_params = {
            'region': region,
            'confidence': debug['ev_probability'],
            'amplitude': debug['ev_amplitude'],
            'confidence_list': debug['confidence_list'],
            'charger_type': debug['charger_type_list']
        }

        if debug['disagg_mode'] != 'mtd' and debug['model_probability'] > 0.5:
            required_params['post_processing_param'] = \
                {
                    'box_monthly_count_var': debug['box_monthly_count_var'],
                    'box_monthly_presence_var': debug['box_monthly_presence_var'],
                    'box_seasonal_count_var': debug['box_seasonal_count_var'],
                    'seasonal_boxes_frac': debug['seasonal_boxes_frac'],
                    'prom_smr_hrs': debug['prom_smr_hrs'],
                    'prom_wtr_hrs': debug['prom_wtr_hrs'],
                    'first_ev_month': debug['first_ev_month'],
                    'last_ev_month': debug['last_ev_month'],
                    'charging_freq': debug['charging_freq'],
                    'charges_per_day': debug['charges_per_day'],
                    'frac_multi_charge': debug['frac_multi_charge']
                }

        ev_heatmaps(required_params, uuid, sampling_rate, pilot, user_dir, logger)

        logger.info('Plots/Heatmaps saved locally | ')

    return


def save_csv(debug, uuid, sampling_rate, out_bill_cycles, timezone, save_path, padding_zero, disagg_mode, logger_base):
    """
    Parameters:
        debug               (dict)      : Algorithm intermediate steps output
        uuid                (str)       : User id
        sampling_rate       (int)       : Sampling rate of the user
        out_bill_cycles     (list)      : List of the bill cycles for which output is to be saved
        timezone            (int)       : Timezone in hours w.r.t. GMT
        save_path           (str)       : Path to save output files
        padding_zero        (bool)      : If the bill cycle boundaries to be of zeros or max
        disagg_mode         (str)       : Disaggregation mode
        logger_base         (logger)    : Logger object
    Return:

    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('save_csv')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve the relevant files from the debug object

    raw = debug['original_input_data']
    input_data = debug['input_data']

    if debug['charger_type'] == 'L1':
        detection = debug['l1']['features_box_data']
    else:
        detection = debug['features_box_data']

    ev_output = debug['final_ev_signal']
    residual = debug['residual_data']
    temp_pot = deepcopy(debug['input_data'][:, Cgbdisagg.INPUT_TEMPERATURE_IDX])
    input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = temp_pot

    # Fill NaNs with zero

    raw[np.isnan(raw)] = 0
    input_data[np.isnan(input_data)] = 0
    detection[np.isnan(detection)] = 0
    ev_output[np.isnan(ev_output)] = 0
    residual[np.isnan(residual)] = 0

    # Cap the negative values with zero

    raw = np.fmax(raw, 0)
    input_data = np.fmax(input_data, 0)
    detection = np.fmax(detection, 0)
    ev_output = np.fmax(ev_output, 0)
    residual = np.fmax(residual, 0)

    # Update current working directory to uuid folder

    current_working_dir = os.getcwd()
    original_pipeline_path = deepcopy(current_working_dir)

    logger.info('Current Working directory | {}'.format(current_working_dir))

    # noinspection PyBroadException
    try:
        # Check if the current working directory changes

        os.chdir(current_working_dir + '/' + save_path)
    except:
        # If the current working directory didn't change, get the default one

        current_working_dir = '/'
        original_pipeline_path = deepcopy(current_working_dir)
        os.chdir(current_working_dir + '/' + save_path)

    logger.info('The new current working directory | {}'.format(current_working_dir + '/' + save_path))

    # Unzip the files (if present in the target location)

    file_unzip(logger)

    # Save the user level features only in historical/incremental mode

    if disagg_mode != 'mtd':
        save_features(debug, disagg_mode, uuid, logger)

    # Save raw data and other EV information as combined output data

    already_present_epochs = save_combined_data(raw, input_data, detection, ev_output, residual, uuid, disagg_mode, logger)

    # Convert the raw data and water heater output to 2d-matrices where each row represents single day

    daily_data_output = get_day_data(raw, input_data, detection, ev_output, residual, timezone, sampling_rate,
                                     padding_zero)

    # Retrieve all the corresponding matrices from the output list

    month_ts, day_ts, epoch_ts, raw_data, input_data, detection_data, ev_data, residual_data = daily_data_output

    # Replace NaN with zero for all the output data tables

    day_ts[np.isnan(day_ts)] = 0
    ev_data[np.isnan(ev_data)] = 0
    month_ts[np.isnan(month_ts)] = 0
    epoch_ts[np.isnan(epoch_ts)] = 0
    raw_data[np.isnan(raw_data)] = 0
    input_data[np.isnan(input_data)] = 0
    residual_data[np.isnan(residual_data)] = 0
    detection_data[np.isnan(detection_data)] = 0

    # Replace Inf / -Inf with zero for all the output data tables

    day_ts[np.isinf(np.abs(day_ts))] = 0
    ev_data[np.isinf(np.abs(ev_data))] = 0
    month_ts[np.isinf(np.abs(month_ts))] = 0
    epoch_ts[np.isinf(np.abs(epoch_ts))] = 0
    raw_data[np.isinf(np.abs(raw_data))] = 0
    input_data[np.isinf(np.abs(input_data))] = 0
    residual_data[np.isinf(np.abs(residual_data))] = 0
    detection_data[np.isinf(np.abs(detection_data))] = 0

    # Declare the file names for all the saved outputs

    ev_csv = uuid + '_ev.csv'
    raw_csv = uuid + '_raw.csv'
    input_csv = uuid + '_input.csv'
    day_ts_csv = uuid + '_day_ts.csv'
    month_ts_csv = uuid + '_month_ts.csv'
    epoch_ts_csv = uuid + '_epoch_ts.csv'
    residual_csv = uuid + '_residual.csv'
    detection_csv = uuid + '_detection.csv'
    confidence_csv = uuid + '_confidence.csv'
    charger_type_csv = uuid + '_charger_type.csv'

    # Check which days data is to be retained

    days_to_include = np.in1d(np.nanmin(month_ts, axis=1), out_bill_cycles)
    epochs_to_include = ~np.in1d(np.nanmin(epoch_ts, axis=1), np.asarray(already_present_epochs))
    days_to_include = days_to_include & epochs_to_include

    # Subset the data based on the required days found above

    ev_copy = ev_data[days_to_include, :].copy()
    raw_copy = raw_data[days_to_include, :].copy()
    day_ts_copy = day_ts[days_to_include, :].copy()
    input_copy = input_data[days_to_include, :].copy()
    month_ts_copy = month_ts[days_to_include, :].copy()
    epoch_ts_copy = epoch_ts[days_to_include, :].copy()
    residual_copy = residual_data[days_to_include, :].copy()
    detection_copy = detection_data[days_to_include, :].copy()

    # Get the number of columns in the 2D matrix

    num_cols = raw_copy.shape[1]

    if os.path.exists(current_working_dir + '/' + save_path + uuid + '_month_ts.csv'):

        temp = pd.read_csv(current_working_dir + '/' + save_path + uuid + '_month_ts.csv', header=None)
        previous_cols = temp.shape[1]

        if previous_cols < num_cols:
            ev_copy = downsample_data(ev_copy, previous_cols, num_cols, method='sum')
            raw_copy = downsample_data(raw_copy, previous_cols, num_cols, method='sum')
            input_copy = downsample_data(input_copy, previous_cols, num_cols, method='sum')
            residual_copy = downsample_data(residual_copy, previous_cols, num_cols, method='sum')
            detection_copy = downsample_data(detection_copy, previous_cols, num_cols, method='sum')
            month_ts_copy = downsample_data(month_ts_copy, previous_cols, num_cols, method='copy')
            day_ts_copy = downsample_data(day_ts_copy, previous_cols, num_cols, method='copy')
            epoch_ts_copy = downsample_data(epoch_ts_copy, previous_cols, num_cols, method='copy')
            month_ts = downsample_data(month_ts, previous_cols, num_cols, method='copy')
            day_ts = downsample_data(day_ts, previous_cols, num_cols, method='copy')
            epoch_ts = downsample_data(epoch_ts, previous_cols, num_cols, method='copy')

            num_cols = previous_cols

    if padding_zero:
        padded_row = np.full(shape=(1, num_cols), fill_value=0)
    else:
        padded_row = np.full(shape=(1, num_cols), fill_value=np.max(raw_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))

    # Add the padded row to the start and end of the data matrix

    ev_copy = np.r_[padded_row, ev_copy, padded_row]
    raw_copy = np.r_[padded_row, raw_copy, padded_row]
    input_copy = np.r_[padded_row, input_copy, padded_row]
    residual_copy = np.r_[padded_row, residual_copy, padded_row]
    detection_copy = np.r_[padded_row, detection_copy, padded_row]

    month_ts_copy = np.r_[np.reshape(month_ts[0, :], newshape=(1, num_cols)), month_ts_copy,
                          np.reshape(month_ts[-1, :], newshape=(1, num_cols))]
    day_ts_copy = np.r_[np.reshape(day_ts[0, :], newshape=(1, num_cols)), day_ts_copy,
                        np.reshape(day_ts[-1, :], newshape=(1, num_cols))]
    epoch_ts_copy = np.r_[np.reshape(epoch_ts[0, :], newshape=(1, num_cols)), epoch_ts_copy,
                          np.reshape(epoch_ts[-1, :], newshape=(1, num_cols))]
    confidence_list = [debug['ev_probability']]
    charger_type_list = [debug['charger_type']]

    # Convert numpy array to data frames

    ev_df = pd.DataFrame(data=ev_copy)
    raw_df = pd.DataFrame(data=raw_copy)
    input_df = pd.DataFrame(data=input_copy)
    day_ts_df = pd.DataFrame(data=day_ts_copy)
    month_ts_df = pd.DataFrame(data=month_ts_copy)
    epoch_ts_df = pd.DataFrame(data=epoch_ts_copy)
    residual_df = pd.DataFrame(data=residual_copy)
    detection_df = pd.DataFrame(data=detection_copy)
    confidence_df = pd.DataFrame(data=confidence_list)
    charger_type_df = pd.DataFrame(data=charger_type_list)

    # Save the data frames to the target location

    ev_df.to_csv(ev_csv, mode='a', header=None, index=None)
    raw_df.to_csv(raw_csv, mode='a', header=None, index=None)
    input_df.to_csv(input_csv, mode='a', header=None, index=None)
    day_ts_df.to_csv(day_ts_csv, mode='a', header=None, index=None)
    month_ts_df.to_csv(month_ts_csv, mode='a', header=None, index=None)
    epoch_ts_df.to_csv(epoch_ts_csv, mode='a', header=None, index=None)
    residual_df.to_csv(residual_csv, mode='a', header=None, index=None)
    detection_df.to_csv(detection_csv, mode='a', header=None, index=None)
    confidence_df.to_csv(confidence_csv, mode='a', header=None, index=None)
    charger_type_df.to_csv(charger_type_csv, mode='a', header=None, index=None)

    confidence_list = pd.read_csv(confidence_csv, header=None).values.flatten()
    debug['confidence_list'] = confidence_list

    charger_type_list = pd.read_csv(charger_type_csv, header=None).values.flatten()
    debug['charger_type_list'] = charger_type_list

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
    Return:
    """

    # Create a new file handle (overwrite existing one too)

    file_handle = open(save_path + uuid + '_debug.pkl', 'wb')

    # Save the debug dict in the file

    pickle.dump(debug_dict, file_handle)

    # Close the file

    file_handle.close()

    return


def save_features(debug, disagg_mode, uuid, logger):
    """
    Parameters:
        debug               (dict)      : Algorithm intermediate steps output
        uuid                (str)       : User id
        disagg_mode         (str)       : Disaggregation mode
        logger         (logger)    : Logger object
    Return:

    """
    overall_features = pd.DataFrame(debug['user_features'], index=[0])
    overall_features.insert(loc=0, column='feature_level', value='Overall')

    recent_features = pd.DataFrame(debug['user_recent_features'], index=[0])
    recent_features.insert(loc=0, column='feature_level', value='Recent')

    features_data = pd.concat([overall_features, recent_features], axis=0)
    features_data.insert(loc=0, column='disagg_mode', value=disagg_mode)

    features_data.to_csv(uuid + '_features.csv', mode='a', index=False)

    logger.info('Features data saved successfully')

    return


def save_combined_data(raw, input_data, detection, ev_output, residual, uuid, disagg_mode, logger):
    """
    This functions saves epoch level ev combined output including estimation, detection, residual, etc.
    Parameters:
        raw                 (np.ndarray)      : 2-d raw data
        input_data          (np.ndarray)      : 2-d EV algo input data
        detection           (np.ndarray)      : 2-d detection boxes
        ev_output           (np.ndarray)      : 2-d EV estimation data
        residual            (np.ndarray)      : 2-d residual data after EV estimation
        uuid                (str)             : User id
        disagg_mode         (str)             : Disaggregation mode
        logger              (logger)          : Logger object
    Return:
    """

    combined_data = deepcopy(raw[:, :Cgbdisagg.INPUT_SUNSET_IDX])

    combined_data[:, 8] = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    combined_data[:, 9] = detection[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    combined_data[:, 10] = ev_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    combined_data[:, 11] = residual[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Define the list of columns for LAP data

    data_columns = ['bill_cycle_ts', 'week_ts', 'day_ts', 'day_of_week', 'hour_of_day', 'epoch_ts',
                    'raw_energy', 'temperature', 'input_data', 'detection_data', 'ev_output', 'residual']

    combined_data = pd.DataFrame(combined_data, columns=data_columns)

    combined_data.insert(loc=0, column='disagg_mode', value=disagg_mode)
    already_present_epochs = []
    # Check if earlier combined data already present

    # noinspection PyBroadException
    try:
        old_combined_data = pd.read_csv(uuid + '_combined_output.csv')

        combined_data = pd.concat([old_combined_data, combined_data], axis=0)

        combined_data.drop_duplicates(subset='epoch_ts', keep='last', inplace=True)
        already_present_epochs = old_combined_data['epoch_ts']
        logger.info('Existing master data combined | ')
    except (OSError, IOError, ValueError, KeyError):
        logger.info('New master data will be saved | ')

    combined_data.to_csv(uuid + '_combined_output.csv', index=False)

    return already_present_epochs


def downsample_data(data, previous_cols, current_cols, method='sum'):
    """
    This function is used to downsample the data
    Args:
        data                (np.ndarray)            : 2d data array
        previous_cols       (int)                   : Number of columns in the previous run
        current_cols        (int)                   : Number of columns in the current run
        method              (string)                : Downsampling aggregation method
    Returns:
        data_copy           (np.ndarray)            : Downsampled data
    """

    downsampling_rate = int(current_cols / previous_cols)

    data_copy = np.full(shape=(data.shape[0], previous_cols), fill_value=0.0)

    if method == 'sum':
        for i in range(0, data.shape[1], downsampling_rate):
            j = int(i / downsampling_rate)
            data_copy[:, j] = np.nansum(data[:, i:i + downsampling_rate], axis=1)

    if method == 'copy':
        for i in range(0, data.shape[1], downsampling_rate):
            j = int(i / downsampling_rate)
            data_copy[:, j] = data[:, i]

    return data_copy
