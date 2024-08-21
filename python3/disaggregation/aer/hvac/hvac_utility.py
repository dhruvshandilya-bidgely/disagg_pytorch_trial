"""
Author - Abhinav Srivastava
Date - 10th Oct 2018
Hold hvac specific utility functions
"""

# Import python packages

import os
import copy
import scipy
import logging
import numpy as np
import pandas as pd

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.plot_monthly_bar import plot_monthly_bar
from python3.disaggregation.aer.hvac.plot_appmap import generate_appliance_heatmap_new
from python3.disaggregation.aer.hvac.write_analytics import write_hvac_analytics
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params


def read_hsm_in(disagg_input_object, logger_hvac):
    """
    Function attempts to read hsm for hvac. If hsm is not found, initializes to none

    Parameters:

        disagg_input_object (dict)              : Contains key input related information
        logger_hvac         (logging object)    : Records log during code flow

    :Returns:
        hsm_in              (dict)              : Contains hvac hsm
    """

    # noinspection PyBroadException
    try:

        # get global hsm dictionary
        hsm_dic = disagg_input_object.get("appliances_hsm", {})

        # get hvac hsm dictionary
        hsm_in = hsm_dic.get("hvac")

        # check for hvac attributes
        hsm_keys = hsm_dic['hvac']['attributes'].keys()

        # override hsm if no valid hsm found
        if ('found' in hsm_keys) or ('newalgo_cddfoun' in hsm_keys) or ('newalgo_hddfound' in hsm_keys):
            logger_hvac.info('Skipping MTD Mode as hsm is from HVAC 2 or MATLAB |', )
            hsm_in = None

    except (KeyError, TypeError):

        # failsafe hsm
        hsm_in = None

    return hsm_in


def write_analytics_month_and_epoch_results(disagg_input_object, disagg_output_object, month_ao_hvac_res_net):
    """
    Function to write analytics attributes, Month level and epoch level hvac results as CSV dumps

    Parameters:
        disagg_input_object (dict)          : Dictionary containing all the inputs
        disagg_output_object (dict)         : Dictionary containing all the outputs
        month_ao_hvac_res_net (np.ndarray)    : Array containing Month-AO-AC-SH-Residue-Net energies

    Returns:
        None
    """

    # dump analytics information if enabled
    if (disagg_input_object['switch']['hvac']['metrics']) and disagg_input_object['config']['disagg_mode'] != 'mtd':
        write_hvac_analytics(disagg_input_object, disagg_output_object['analytics']['values'], disagg_output_object)

    # dump epoch level estimate if enabled
    if disagg_input_object['switch']['hvac']['epoch_estimate'] and \
            disagg_input_object['config']['disagg_mode'] != 'mtd':
        write_epoch_hvac(disagg_input_object, disagg_output_object)

    # dump month level estimate if enabled
    if disagg_input_object['switch']['hvac']['month_estimate'] and \
            disagg_input_object['config']['disagg_mode'] != 'mtd':

        consumption_frame = pd.DataFrame(month_ao_hvac_res_net,
                                         columns=['Month', 'day_ao', 'ac_od', 'sh_od', 'residue', 'net'])
        month_baseload = disagg_output_object['ao_seasonality']['baseload'] / Cgbdisagg.WH_IN_1_KWH
        month_ac_ao = disagg_output_object['ao_seasonality']['cooling'] / Cgbdisagg.WH_IN_1_KWH
        month_sh_ao = disagg_output_object['ao_seasonality']['heating'] / Cgbdisagg.WH_IN_1_KWH

        consumption_frame['bl'] = month_baseload
        consumption_frame['ac_ao'] = month_ac_ao
        consumption_frame['sh_ao'] = month_sh_ao
        consumption_frame['grey'] = consumption_frame['day_ao'] - (consumption_frame['ac_ao'] +
                                                                   consumption_frame['sh_ao'] + consumption_frame['bl'])

        user_monthly_hvac_folder = os.path.join('../', "monthly_hvac")

        if not os.path.exists(user_monthly_hvac_folder):
            os.makedirs(user_monthly_hvac_folder)

        consumption_frame.to_csv(user_monthly_hvac_folder + '/' + disagg_input_object['config']['uuid'] + '.csv',
                                 index=False)


def write_ao_od_hvac_at_epoch(disagg_input_object, disagg_output_object, epoch_ao_hvac_true_backup):
    """
    Function to dump epoch level HVAC results where On demand and AO components are dumped separately

    Parameters:
        disagg_input_object         (dict)          : Dictionary containing all the inputs
        disagg_output_object        (dict)          : Dictionary containing all the outputs
        epoch_ao_hvac_true_backup   (np.ndarray)    : 2D array

    Returns:
        None
    """

    static_params = hvac_static_params()

    config = disagg_input_object.get('config')

    # dump consumption information if enabled
    if disagg_input_object['switch']['hvac']['epoch_od_ao_hvac_dump'] and \
            disagg_input_object['config']['disagg_mode'] != 'mtd':

        # CSV dump for residential and SMB Separately
        if config.get('user_type').lower() != 'smb':
            # initializing consumption frame
            consumption_frame = pd.DataFrame(epoch_ao_hvac_true_backup,
                                             columns=['epoch', 'baseload', 'ac_demand', 'sh_demand'])

            # adding ao hvac components
            consumption_frame['ac_ao'] = disagg_output_object['ao_seasonality']['epoch_cooling']
            consumption_frame['sh_ao'] = disagg_output_object['ao_seasonality']['epoch_heating']

            # adding raw consumption and temperature
            consumption_frame['temperature'] = disagg_input_object['input_data'][:, Cgbdisagg.INPUT_TEMPERATURE_IDX]
            consumption_frame['net_energy'] = disagg_input_object['input_data'][:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

            # dump folder
            user_epoch_hvac_folder = static_params.get('path').get('epoch_hvac_dir')

            if not os.path.exists(user_epoch_hvac_folder):
                os.makedirs(user_epoch_hvac_folder)

            # dump results
            consumption_frame.to_csv(user_epoch_hvac_folder + '/' + disagg_input_object['config']['uuid'] + '.csv',
                                     index=False)
        else:

            # prepare consumption frame
            consumption_frame = pd.DataFrame(disagg_output_object['epoch_estimate'],
                                             columns=['epoch', 'baseload', 'ac', 'sh', 'extra_ao', 'ac_open',
                                                      'ac_close', 'sh_open',
                                                      'sh_close', 'operational'])

            # add temperature and energy in frame
            consumption_frame['open_flag'] = disagg_input_object['switch']['smb']['open_close_table_write_epoch']
            consumption_frame['temperature'] = disagg_input_object['input_data'][:, Cgbdisagg.INPUT_TEMPERATURE_IDX]
            consumption_frame['net_energy'] = disagg_input_object['input_data'][:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

            # add bill cycle info
            all_bill_cycles = disagg_input_object['input_data'][:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]
            out_bill_cycles = disagg_input_object['out_bill_cycles'][:, 0]
            out_epochs = np.isin(all_bill_cycles, out_bill_cycles)

            out_frame = consumption_frame[out_epochs]

            # getting dump folder
            user_epoch_hvac_folder = static_params.get('path').get('epoch_hvac_dir')

            if not os.path.exists(user_epoch_hvac_folder):
                os.makedirs(user_epoch_hvac_folder)

            # dumping csv

            uuid = config.get('uuid')
            disagg_mode = config.get('disagg_mode')
            bc_start = str(out_bill_cycles[0])
            bc_end = str(out_bill_cycles[-1])
            extension = '.csv'

            energy_file_location = user_epoch_hvac_folder + '/' + uuid + '_' + disagg_mode + '_' + bc_start + '_' + bc_end + extension

            out_frame.to_csv(energy_file_location, header=False, index=False)

            # getting smb open-close info
            open_close_table = disagg_input_object['switch']['smb']['open_close_table_write_day']

            number_of_out_days = len(np.unique(disagg_input_object['input_data'][out_epochs, Cgbdisagg.INPUT_DAY_IDX]))
            open_close_table_out = open_close_table.tail(number_of_out_days)

            # getting info of open close timings
            user_open_close_folder = static_params.get('path').get('open_close_data')

            if not os.path.exists(user_open_close_folder):
                os.makedirs(user_open_close_folder)

            work_hour_file_location = user_open_close_folder + '/' + uuid + '_' + disagg_mode + '_' + bc_start + '_' + bc_end + extension

            # dumping open close info
            open_close_table_out.to_csv(work_hour_file_location, header=False, index=False)


def bar_appmap_baseline(generate_plot, disagg_input_object, disagg_output_object, month_ao_hvac_res_net, column_index,
                        epoch_ao_hvac_true):
    """
    Function to postprocess hvac results in case of over/under estimation, except in mtd mode

    Parameters:

        generate_plot           (bool)          : Boolean flag indicating if the plots has to be generated or not
        disagg_input_object     (dict)          : Dictionary containing all input attributes
        disagg_output_object    (dict)          : Dictionary containing all output attributes
        month_ao_hvac_res_net   (np.ndarray)    : Array containing | month-ao-ac-sh-residue-net energies
        column_index            (dict)          : Dictionary containing column identifier indices of ao-ac-sh
        epoch_ao_hvac_true      (np.ndarray)    : Array containing | epoch-ao-ac-sh energies

    Returns:
    """

    # plot monthly bar if enabled
    if generate_plot:
        if disagg_input_object['switch']['plot_level'] >= 1:
            plot_monthly_bar(disagg_input_object, disagg_output_object, month_ao_hvac_res_net, column_index, 'true')

    # plot heatmap if enabled
    if generate_plot:
        if (disagg_input_object['switch']['plot_level'] >= 3) and disagg_input_object['config']['disagg_mode'] != 'mtd':
            generate_appliance_heatmap_new(disagg_input_object, disagg_output_object, epoch_ao_hvac_true,
                                           column_index, 'true')


def write_epoch_hvac(disagg_input_object, disagg_output_object):
    """
    Function to write epoch level hvac estimates

    Parameters:

        disagg_input_object     (dict)     : Dictionary containing all the disagg related inputs
        disagg_output_object    (dict)     : Dictionary containing all the disagg related outputs

    Returns:
    """

    # getting uuid
    uuid = disagg_input_object['config']['uuid']

    # getting key appliance consumption at epoch level
    ao_baseload = np.around(disagg_output_object['ao_seasonality']['epoch_baseload'], 1)
    ao_cooling = np.around(disagg_output_object['ao_seasonality']['epoch_cooling'], 1)
    ao_heating = np.around(disagg_output_object['ao_seasonality']['epoch_heating'], 1)
    ao_grey = np.around(disagg_output_object['ao_seasonality']['epoch_grey'], 1)
    ac_consumption = np.around(disagg_output_object['epoch_estimate'][:, 2], 1)
    sh_consumption = np.around(disagg_output_object['epoch_estimate'][:, 3], 1)

    # getting raw info at epoch level
    input_data = disagg_input_object['input_data']
    epoch_input_data = copy.deepcopy(input_data)
    epoch_time = epoch_input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]
    epoch_temperature = np.around(epoch_input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX], 1)
    epoch_net = epoch_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # preparing epoch frame
    epoch_hvac = np.c_[
        epoch_time, epoch_net, ao_baseload, ao_cooling, ao_heating, ao_grey, ac_consumption, sh_consumption, epoch_temperature]
    column_names = ['epoch', 'net_energy', 'ao_baseload', 'ao_cooling', 'ao_heating', 'ao_grey', 'ac_on_demand',
                    'sh_on_demand', 'temperature']
    df_epoch_hvac = pd.DataFrame(epoch_hvac, columns=column_names)

    # initializing dump directory
    epoch_hvac_dir = os.path.join('../', "epoch_hvac")
    if not os.path.exists(epoch_hvac_dir):
        os.makedirs(epoch_hvac_dir)

    # dumping epoch estimates
    pd.DataFrame.to_csv(df_epoch_hvac, epoch_hvac_dir + '/' + uuid + '.csv', header=False, index=False)


def get_residue_stability(residual, qualified_residual, logger_base):
    """
    Function to get the stability of month level residuals

    Parameters:
        residual            (np.ndarray)     : Array containing all the monthly residuals
        qualified_residual  (np.ndarray)     : Array containing all the qualified monthly residuals
        logger_base         (logging object) : Records code flow logs

    Returns:
        stability_metric    (float)          : Contains extent od residual stability
    """

    # Initiate logger for the hvac module
    logger_hvac_base = logger_base.get('logger').getChild('get_res_stability')
    logger_get_res_stability = logging.LoggerAdapter(logger_hvac_base, logger_base.get('logging_dict'))

    static_params = hvac_static_params()

    # initializing consumption frame
    df = pd.DataFrame()

    logger_get_res_stability.info("getting valid residuals |")

    # getting residual added
    residual = residual[qualified_residual == True]
    residual = residual // static_params['residual_quantization_val']
    df['residual'] = np.around(residual)

    # getting stability metrics
    if sum(df['residual']) == 0:

        stability_metric = (1, static_params['default_residual_stability'])
        logger_get_res_stability.info("stability metric is {} |".format(stability_metric))

    else:
        df['residual_scaled'] = np.around(df['residual'] / np.median(df['residual'][df['residual'] > 0]), 2)
        stability_metric = (np.around(np.mean(df['residual'][df['residual'] > 0]), 2),
                            np.around(np.std(df['residual'][df['residual'] > 0]), 2))

        logger_get_res_stability.info("stability metric is {} |".format(stability_metric))

    return stability_metric


def quantize_month_degree_day(cdd_months, month_cdd, hdd_months, month_hdd):
    """
    Function to quantize month level measure of degrees for cooling and heating

    Parameters:
        cdd_months          (np.ndarray)    : Array containing information about validity of a month for cooling
        month_cdd           (np.ndarray)    : Array containing information about cdd at month level, 65 base
        hdd_months          (np.ndarray)    : Array containing information about validity of a month for heating
        month_hdd           (np.ndarray)    : Array containing information about hdd at month level, 65 base

    Returns:
        month_cdd_quantized (np.ndarray)    : Array containing quantized cdd at month level
        month_hdd_quantized (np.ndarray)    : Array containing quantized hdd at month level
    """

    static_params = hvac_static_params()

    # getting month cdd-hdd quantized initiated
    month_cdd_quantized = np.array([])
    month_hdd_quantized = np.array([])

    valid = 1

    # getting month cdd quantized
    if sum(cdd_months) >= static_params['len_residual_round']:

        # noinspection PyBroadException
        try:
            month_cdd_quantized = (month_cdd - np.min(month_cdd[cdd_months == valid])) / np.std(
                month_cdd[cdd_months == valid])
        except (ValueError, IndexError):
            month_cdd_quantized = month_cdd

    # getting month hdd quantized
    if sum(hdd_months) >= static_params['len_residual_round']:
        try:
            month_hdd_quantized = (month_hdd - np.min(month_hdd[hdd_months == valid])) / np.std(
                month_hdd[hdd_months == valid])
        except (ValueError, IndexError):
            month_hdd_quantized = month_hdd

    return month_cdd_quantized, month_hdd_quantized


def get_extreme_temp_days(hvac_input_temperature, unique_day_idx, day_idx, invalid_idx, config, season_label):
    """
    Function to extract extreme (hottest/coldest) temperature days from the entire input data
    Args:
        hvac_input_temperature  (np.ndarray)    : Array of epoch level temperature flowing into hvac module
        unique_day_idx          (np.ndarray)    : Array of Indices of unique days
        day_idx                 (np.ndarray)    : Array of epoch level day indices
        invalid_idx             (np.ndarray)    : Array of invalid epochs based on consumption and temperature
        config                  (dict)          : Dictionary with fixed parameters to calculate user characteristics
        season_label            (str)           : String identifying summer or winter

    Returns:
        extreme_temp_bool       (np.ndarray)    : Array of boolean values marking most n extreme temperature days
    """
    # Initialise
    ndays = min(config['extreme_days'], len(unique_day_idx))
    hvac_input_temperature = copy.deepcopy(hvac_input_temperature)
    hvac_input_temperature[invalid_idx == 1] = 0
    extreme_temp_bool = np.zeros(hvac_input_temperature.shape).astype(int)

    valid_hours = np.where(hvac_input_temperature > 0, 1, 0)

    # Calculate mean daily temperature
    aggregate_day_temp = np.bincount(day_idx, hvac_input_temperature)
    count_valid_hours_daily = np.bincount(day_idx, valid_hours)
    mean_day_temp = aggregate_day_temp / count_valid_hours_daily

    # Suppress mean to zero on invalid days
    mean_day_temp = np.where(np.isnan(mean_day_temp) | np.isinf(mean_day_temp), 0, mean_day_temp)

    # Flag hottest bill cycle days for summer
    if season_label == 'summer':
        sorted_day_idx = np.argsort(mean_day_temp)[::-1][0:ndays]
        extreme_temp_lower_cutoff = mean_day_temp[sorted_day_idx[-1]]

        mean_day_temp = mean_day_temp[day_idx]
        hottest_days_bool = np.where(mean_day_temp >= extreme_temp_lower_cutoff, 1, 0)
        extreme_temp_bool = hottest_days_bool

    # Flag coldest bill cycle days for winter
    elif season_label == 'winter':
        exclude_days_invalid_temp = int(np.sum(np.where(mean_day_temp == 0, 1, 0)))
        sorted_day_idx = np.argsort(mean_day_temp)[exclude_days_invalid_temp: exclude_days_invalid_temp + ndays]
        extreme_temp_upper_cutoff = mean_day_temp[sorted_day_idx[-1]]

        mean_day_temp = mean_day_temp[day_idx]
        coldest_days_bool = np.where(mean_day_temp <= extreme_temp_upper_cutoff, 1, 0)
        extreme_temp_bool = coldest_days_bool

    return extreme_temp_bool


def get_all_indices(hvac_input_data, invalid_idx, config):
    """
    Function to extract and store all epoch/hour/day/month level indices in one object
    Args:
        hvac_input_data     (np.ndarray)    : 2D array of epoch level consumption and temperature data
        invalid_idx         (np.ndarray)    : Array of invalid epochs based on consumption and temperature
        config              (dict)          : Dictionary with fixed parameters to calculate user characteristics

    Returns:
        all_indices         (dict)          : Dictionary containing all day/epoch/month/hour level indices
    """
    # Copy and read inputs
    hvac_input_data = copy.deepcopy(hvac_input_data)

    hvac_input_temperature = hvac_input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX].squeeze()
    hvac_input_consumption = hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX].squeeze()
    s_labels = hvac_input_data[:, Cgbdisagg.INPUT_S_LABEL_IDX]

    invalid_idx = np.logical_or(invalid_idx,
                                np.logical_or(np.isnan(hvac_input_temperature), np.isnan(hvac_input_consumption)))
    hvac_input_temperature[invalid_idx == 1] = np.nan
    hvac_input_consumption[invalid_idx == 1] = np.nan

    # Calculate hour , day and month (bill cycle) indices for references throughout the remaining code
    _, unique_hour_idx, hour_idx = scipy.unique(hvac_input_data[:, Cgbdisagg.INPUT_HOD_IDX], return_index=True,
                                                return_inverse=True)
    _, unique_day_idx, day_idx = scipy.unique(hvac_input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_index=True,
                                              return_inverse=True)
    _, unique_month_idx, month_idx = scipy.unique(hvac_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                                                  return_index=True, return_inverse=True)

    # Calculate boolean arrays flagging extreme temperature bill cycle epochs
    hottest_day_bool = get_extreme_temp_days(hvac_input_temperature, unique_day_idx, day_idx, invalid_idx, config,
                                             'summer')
    coldest_day_bool = get_extreme_temp_days(hvac_input_temperature, unique_day_idx, day_idx, invalid_idx, config,
                                             'winter')

    # Store all indices
    all_indices = {
        'unique_day_idx': unique_day_idx,
        'day_idx': day_idx,
        'invalid_idx': invalid_idx,
        'unique_month_idx': unique_month_idx,
        'month_idx': month_idx,
        'hour_idx': hour_idx,
        'unique_hour_idx': unique_hour_idx,
        's_labels': s_labels,
        'hottest_day_bool': hottest_day_bool,
        'coldest_day_bool': coldest_day_bool
    }

    return all_indices


def max_min_without_inf_nan(arr, operation='max'):
    """
    Utility function to take maximum and minimum excluding nan or inf values

    Parameters:
        arr         (list)          : Input nested array/list of numbers
        operation   (str)           : String identifier for max or min operation

    Returns:
        output      (float)         : Overall max / min of the input arr
    """
    arr = np.array(arr)
    valid_values = arr[(arr != np.Inf) & ~(np.isnan(arr))]
    output = np.nan
    if valid_values.size == 0:
        return output
    elif operation == 'max':
        output = np.max(valid_values)
        return output
    output = np.min(valid_values)
    return output


def softmax_custom(logit):
    """
    Utility function to calculated softmax probability from logit
    Arguments:
        logit           (np.ndarray)    : Input logit value which is the dot product of features
    Return:
        softmax_output  (float)         : Output of softmax function
    """
    softmax_output = 1 / (1 + np.exp(-1 * logit))
    return softmax_output
