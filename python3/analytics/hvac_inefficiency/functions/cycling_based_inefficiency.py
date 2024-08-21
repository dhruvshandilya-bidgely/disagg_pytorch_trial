"""
Author  -   Anand Kumar Singh
Date    -   22th Feb 2021
This file contains code for finding cycling based inefficiencies in HVAC consumption
"""

# Import python packages

import copy
import logging
import numpy as np

# Import functions from within the project

from python3.utils.maths_utils.matlab_utils import percentile_1d
from python3.analytics.hvac_inefficiency.functions.saturation_functions import get_saturation
from python3.analytics.hvac_inefficiency.functions.hvac_clustering import correct_hvac_cluster
from python3.analytics.hvac_inefficiency.functions.duty_cycle import compute_scaled_duty_cycle
from python3.analytics.hvac_inefficiency.functions.duty_cycle import compute_absolute_duty_cycle
from python3.analytics.hvac_inefficiency.functions.saturation_functions import get_pre_saturation
from python3.analytics.hvac_inefficiency.functions.hvac_clustering import cluster_hvac_consumption
from python3.analytics.hvac_inefficiency.utils.continuous_consumption import get_continuous_low_consumption
from python3.analytics.hvac_inefficiency.functions.get_short_cycling_streaks import get_short_cycling_streaks
from python3.analytics.hvac_inefficiency.configs.init_cycling_based_config import get_cycling_based_ineff_config


def get_device_temp_frame(device, config, dataframe, temperature_column):

    """
    Function to get device temperature and dataframe

    Parameters:
        device              (str)           : HVAC device
        config              (dict)          : Dictionary of config
        dataframe           (object)        : Dataframe
        temperature_column  (int)           : Temperature column

    Returns:
        cut_off_temperature (int)           : Temperature
        dataframe           (object)        : Dataframe
    """

    cut_off_temperature = None

    if device == 'ac':
        cut_off_temperature = config.get('low_ac_temperature')
        dataframe = dataframe[dataframe[:, temperature_column] >= cut_off_temperature]

    elif device == 'sh':
        cut_off_temperature = config.get('high_sh_temperature')
        dataframe = dataframe[dataframe[:, temperature_column] <= cut_off_temperature]

    return cut_off_temperature, dataframe


def encode_temperature(temperature):

    """
    Function to encode temperature

    Parameters:
        temperature     (object)    : Saturation or pre saturation temperature

    Returns:
        temperature     (object)    : Saturation or pre saturation temperature
    """

    if temperature == None:
        temperature = 'None'

    if type(temperature) == int:
        temperature = str(temperature)

    return temperature


def decode_temperature(temperature):

    """
    Function to encode temperature

    Parameters:
        temperature     (str)    : Saturation or pre saturation temperature

    Returns:
        temperature     (object)    : Saturation or pre saturation temperature
    """

    if temperature == 'None':
        temperature = None

    return temperature


def cycling_based_ineff(input_hvac_inefficiency_object, output_hvac_inefficiency_object, logger_pass,
                        device='none'):
    """
        This function estimates cycling based inefficiency

        Parameters:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            output_hvac_inefficiency_object     (dict)          dictionary containing all output information
            logger_pass                         (object)        logger object
            device                              (str)           string indicating device, either AC or SH
        Returns:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            output_hvac_inefficiency_object     (dict)          dictionary containing all output information
    """

    # Initializing logger function

    logger_local = logger_pass.get("logger").getChild("cycling_based_ineff")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    logger.info('Starting Cycling Based HVAC inefficiency |')

    raw_input_data = input_hvac_inefficiency_object.get('raw_input_values')

    demand_hvac_col_idx = input_hvac_inefficiency_object.get('raw_input_idx').get('{}_demand'.format(device))
    ao_hvac_col_idx = input_hvac_inefficiency_object.get('raw_input_idx').get('{}_ao'.format(device))
    temperature_col_idx = input_hvac_inefficiency_object.get('raw_input_idx').get('temperature')
    cons_col_idx = input_hvac_inefficiency_object.get('raw_input_idx').get('consumption')

    # Get data from raw data array

    hvac_consumption =\
        copy.deepcopy(raw_input_data[:, demand_hvac_col_idx] + raw_input_data[:, ao_hvac_col_idx]).reshape(-1, 1)
    temperature = copy.deepcopy(raw_input_data[:, temperature_col_idx]).reshape(-1, 1)
    total_consumption = copy.deepcopy(raw_input_data[:, cons_col_idx]).reshape(-1, 1)

    sampling_rate = input_hvac_inefficiency_object.get('sampling_rate')
    logger.debug('Users sampling rate={} | device={}'.format(sampling_rate, device))

    # Get config for cycling based inefficiencies module

    config = get_cycling_based_ineff_config(input_hvac_inefficiency_object, device)

    logger.debug('Successfully fetched config cycling based config |')

    input_hvac_inefficiency_object[device]['config'] = config

    unrolled_hvac_consumption = copy.deepcopy(hvac_consumption)

    # No HVAC consumption in less than 1.5 hours is considered as off device and others are
    # considered as off compressor cycles.

    allowed_off_time = config.get('max_continuous_off_time')
    unrolled_hvac_consumption =\
        get_continuous_low_consumption(unrolled_hvac_consumption, length=allowed_off_time, min_value=0,
                                       fill_value=np.nan)

    logger.debug('Filtering out %s off region with more than allowed off time | %.2f', device, allowed_off_time)

    valid_hvac_consumption_array = copy.deepcopy(unrolled_hvac_consumption)

    nan_idx = np.isnan(unrolled_hvac_consumption)

    input_hvac_inefficiency_object['hvac_nan_idx'] = nan_idx

    temp_array = unrolled_hvac_consumption[~nan_idx]

    upper_value_hvac_consumption_percentile = config.get('upper_value_hvac_consumption_percentile')

    upper_value_hvac_consumption = np.percentile(temp_array, upper_value_hvac_consumption_percentile)

    logger.debug('Filtering outlier HVAC consumption, percentile=%.1f, consumption|%.1f',
                 upper_value_hvac_consumption_percentile, upper_value_hvac_consumption)

    temp_array[temp_array > upper_value_hvac_consumption] = upper_value_hvac_consumption

    input_hvac_inefficiency_object['hvac_consumption_array'] = temp_array.reshape(-1, 1)

    input_hvac_inefficiency_object, output_hvac_inefficiency_object =\
        cluster_hvac_consumption(input_hvac_inefficiency_object, output_hvac_inefficiency_object, logger_pass, device)

    logger.debug('Finished clustering HVAC consumption |')

    # Cluster correction based on consumption limits per data point

    input_hvac_inefficiency_object, output_hvac_inefficiency_object =\
        correct_hvac_cluster(input_hvac_inefficiency_object, output_hvac_inefficiency_object, logger_pass, device)
    logger.debug('Finished correcting HVAC consumption |')

    device_information = output_hvac_inefficiency_object.get(device)
    predicted_clusters = device_information.get('updated_cluster_information').get('predicted_clusters')
    original_predicted_cluster = copy.deepcopy(predicted_clusters)
    initial_clustering_information = output_hvac_inefficiency_object.get(device).get('initial_clustering_information')
    largest_cluster_id = device_information.get('updated_cluster_information').get('largest_cluster_id')
    smallest_cluster_id = device_information.get('updated_cluster_information').get('smallest_cluster_id')
    second_smallest_cluster_id = device_information.get('updated_cluster_information').get('second_smallest_cluster_id')
    cluster_count = device_information.get('updated_cluster_information').get('cluster_count')

    logger.debug('Final clustering information cluster_count| {}'.format(cluster_count))

    unrolled_hvac_consumption[unrolled_hvac_consumption < 0] = np.nan

    # set lowest cluster id to 0

    upper_cut_of_small_cluster = hvac_consumption[predicted_clusters == smallest_cluster_id].max()

    hvac_consumption[hvac_consumption < upper_cut_of_small_cluster] = 0
    unrolled_hvac_consumption[unrolled_hvac_consumption < upper_cut_of_small_cluster] = 0

    # dealing with largest cluster, setting second largest and below consumption to 0

    single_cluster = 1
    two_clusters = 2
    three_clusters = 3

    if cluster_count == single_cluster:
        unrolled_data_largest_cluster = np.zeros_like(unrolled_hvac_consumption, dtype=float)
    else:
        unrolled_data_largest_cluster = copy.deepcopy(unrolled_hvac_consumption)
        lower_cut_off_largest_cluster = hvac_consumption[predicted_clusters != largest_cluster_id].max()
        unrolled_data_largest_cluster[unrolled_data_largest_cluster < lower_cut_off_largest_cluster] = 0

    # Code to restrict values for short cycling.

    cluster_count = len(np.unique(predicted_clusters))

    if cluster_count == two_clusters:
        nan_idx = np.isnan(unrolled_data_largest_cluster)
        zero_data_idx = unrolled_data_largest_cluster == 0
        valid_idx = (~nan_idx) & (~zero_data_idx)
        valid_consumption = unrolled_data_largest_cluster[valid_idx]
        std_deviation = np.nanstd(valid_consumption)
        median = np.nanmedian(valid_consumption)
        upper_cluster_limit = (median + std_deviation)
        unrolled_data_largest_cluster[unrolled_data_largest_cluster < upper_cluster_limit] = 0

    # Taking partial cycle into account.

    compressor_consumption = np.empty(0,)
    full_cycle_consumption = np.nan

    if cluster_count == single_cluster:
        compressor_consumption = hvac_consumption[predicted_clusters == smallest_cluster_id]
        full_cycle_consumption = np.nanmedian(compressor_consumption) + np.nanstd(compressor_consumption)
    elif cluster_count == two_clusters:
        compressor_cut_off_percentile = config.get('compressor_cut_off_percentile_2')
        compressor_consumption = hvac_consumption[predicted_clusters == second_smallest_cluster_id]
        full_cycle_consumption = percentile_1d(compressor_consumption, compressor_cut_off_percentile)
    elif cluster_count >= three_clusters:
        compressor_cut_off_percentile = config.get('compressor_cut_off_percentile_3')
        compressor_consumption = hvac_consumption[predicted_clusters == largest_cluster_id]
        full_cycle_consumption = percentile_1d(compressor_consumption, compressor_cut_off_percentile)

    logger.debug('Final clustering information FCC value| {:3f}'.format(full_cycle_consumption))

    # Updating HVAC compressor size nothing falls in compressor range

    if compressor_consumption.shape[0] == 0:
        compressor_cut_off_percentile = config.get('compressor_cut_off_percentile_no_comp')
        non_zero_idx = ~(hvac_consumption == 0)
        full_cycle_consumption = percentile_1d(hvac_consumption[non_zero_idx], compressor_cut_off_percentile)
        logger.debug('Final clustering information, zero comp consumption FCC value| {}'.format(full_cycle_consumption))

    # Compute HVAC duty cycle over 3 hours period

    window_length = config.get('scaled_duty_cycle_window')

    duty_cycle_mode_1_array =\
        compute_scaled_duty_cycle(unrolled_hvac_consumption, full_cycle_consumption, window_length=window_length)

    logger.debug('Compute Scaled duty cycle with window length | {}'.format(window_length))

    # Computing duty cycle for largest AC consumption array.

    window_length = config.get('largest_duty_cycle_window_length')

    duty_cycle_array_largest = compute_absolute_duty_cycle(unrolled_data_largest_cluster, window_length)
    logger.debug('Compute largest duty cycle with window length | {}'.format(window_length))

    # Trying to find saturation temperature

    temperature = temperature.ravel()

    # finding indices where temperature and duty cycle are valid numbers

    val_idx = (~np.isnan(duty_cycle_mode_1_array)) & (~np.isnan(temperature))

    val_temp = temperature[val_idx].astype(int)
    val_dcma = duty_cycle_mode_1_array[val_idx].astype(float)

    un_temp, un_temp_idx = np.unique(val_temp, return_inverse=True)

    med_arr = []
    len_arr = []

    logger.debug('Finding median and support for each temperature |')

    for idx in range(len(un_temp)):
        val_data = val_dcma[un_temp_idx == idx]
        med_arr.append(np.median(val_data))
        len_arr.append(len(val_data))

    # Initialising column indices for new numpy array

    temperature_column = 0
    median_dc_column = 1
    length_dc_column = 2

    dataframe = np.c_[un_temp, np.array(med_arr), np.array(len_arr)]
    dataframe = dataframe[~(dataframe[:, temperature_column] == 0)]

    logger.debug('Done creating array with duty cycle relationship |')

    logger.debug('Filtering low/high temperature points for duty cycle relationship')

    cut_off_temperature, dataframe = get_device_temp_frame(device, config, dataframe, temperature_column)

    dataframe_master = copy.deepcopy(dataframe)

    # minimum 20% of the median days epochs required.

    perc_min_data = config.get('perc_min_data_for_relation_curve')
    minimum_data_days = np.median(dataframe[:, length_dc_column]) * perc_min_data

    logger.debug('Filtered minimum data days |')

    dataframe = dataframe[dataframe[:, length_dc_column] > minimum_data_days]

    x = dataframe[:, temperature_column]
    y = dataframe[:, median_dc_column]

    # Initialise saturation temperature and pre saturation fraction

    saturation_fraction = 0
    saturation_temperature = None
    pre_saturation_fraction = 0

    # Saturation condition on duty cycle and make decision

    min_duty_cycle = config.get('pre_sat_min_duty_cycle')
    average_duty_cycle = config.get('pre_sat_average_duty_cycle')

    # Check condition for always high pre saturation

    min_representing_percentile = config.get('min_representing_percentile')

    always_high_pre_saturation =\
        (percentile_1d(y, min_representing_percentile) > min_duty_cycle) and (np.nanmedian(y) > average_duty_cycle)

    if always_high_pre_saturation:
        pre_saturation_temperature = 'Always high'
        pre_saturation_fraction = 1

    elif x.shape[0] == 0:
        pre_saturation_temperature = 'No meaningful {} found'.format(device)

    else:

        pre_saturation_temperature = get_pre_saturation(x, y, min_duty_cycle, average_duty_cycle, logger_pass, device)

        logger.debug('Computed Pre saturation temperature {}'.format(pre_saturation_temperature))

    # Add pre saturation information

    min_duty_cycle = config.get('sat_min_duty_cycle')
    average_duty_cycle = config.get('sat_average_duty_cycle')

    pre_saturation_temperature = encode_temperature(pre_saturation_temperature)

    # If pre saturation temperature detected then find pre saturation fraction

    if len(pre_saturation_temperature) <= 3:

        pre_saturation_temperature = int(pre_saturation_temperature)

        # Adding pre saturation region

        valid_idx = dataframe_master[:, temperature_column] > pre_saturation_temperature

        pre_saturation_region = np.sum(dataframe_master[valid_idx][:, length_dc_column])

        total_region = np.sum(dataframe_master[:, length_dc_column])

        pre_saturation_fraction = pre_saturation_region / total_region

        pre_saturation_fraction = np.round(pre_saturation_fraction, 3)

        if device == 'sh':
            pre_saturation_fraction = 1 - pre_saturation_fraction

        logger.info('Computed pre saturation fraction = {} and temperature | {}'.format(pre_saturation_fraction,
                                                                                        pre_saturation_temperature))

        # Get Saturation Temperature

        saturation_temperature =\
            get_saturation(x, y, min_duty_cycle, average_duty_cycle, pre_saturation_temperature, logger_pass, device)

        logger.debug('Computed saturation temperature | {}'.format(saturation_temperature))

    pre_saturation_temperature = decode_temperature(pre_saturation_temperature)

    # If saturation temperature detected then find saturation fraction

    saturation_temperature = encode_temperature(saturation_temperature)

    if len(saturation_temperature) <= 3:

        saturation_temperature = int(saturation_temperature)

        saturation_region =\
            np.sum(dataframe_master[dataframe_master[:, temperature_column] > saturation_temperature][:, length_dc_column])

        total_region = np.sum(dataframe_master[:, length_dc_column])
        saturation_fraction = saturation_region / total_region
        if device == 'sh':
            saturation_fraction = 1 - saturation_fraction
        logger.info('Computed saturation fraction = {} and temperature | {}'.format(saturation_fraction,
                                                                                    saturation_temperature))

    saturation_temperature = decode_temperature(saturation_temperature)

    # Getting short cycling data points

    short_cycling_array = copy.deepcopy(duty_cycle_array_largest)

    short_cycling_cut_off_freq = config.get('min_duty_cycle_short_cycling')

    # Set short cycling less than cut off freq at zero

    short_cycling_array[short_cycling_array < short_cycling_cut_off_freq] = 0

    # Setting nans as zero

    short_cycling_array[np.isnan(short_cycling_array)] = 0

    short_cycling_array, average_energy_short_cycling_streaks =\
        get_short_cycling_streaks(short_cycling_array, total_consumption, sampling_rate)

    updated_cluster_information =\
        output_hvac_inefficiency_object.get(device).get('updated_cluster_information', dict({}))

    cycling_debug_dictionary = {
        'short_cycling': short_cycling_array,
        'duty_cycle_relationship': dataframe,
        'compressor': unrolled_hvac_consumption,
        'saturation_temp': saturation_temperature,
        'saturation_fraction': saturation_fraction,
        'full_cycle_consumption': full_cycle_consumption,
        'duty_cycle_mode_1_array': duty_cycle_mode_1_array,
        'pre_saturation_fraction': pre_saturation_fraction,
        'duty_cycle_array_largest': duty_cycle_array_largest,
        'pre_saturation_temperature': pre_saturation_temperature,
        'average_sc_energy': average_energy_short_cycling_streaks,
        'unrolled_data_largest_cluster': unrolled_data_largest_cluster,
        'updated_cluster_information': updated_cluster_information.get('cluster_information'),
        'all_cluster_information': updated_cluster_information,
        'original_predicted_clusters': original_predicted_cluster,
        'predicted_clusters': predicted_clusters,
        'valid_hvac': valid_hvac_consumption_array,
        'initial_clustering_information': initial_clustering_information
    }

    output_hvac_inefficiency_object[device]['cycling_debug_dictionary'] = cycling_debug_dictionary

    return input_hvac_inefficiency_object, output_hvac_inefficiency_object
