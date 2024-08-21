"""
Author - Abhinav
Date -
"""

# import python packages
import logging
import copy
import numpy as np
import pandas as pd

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aes.hvac.init_hourly_smb_hvac_params import hvac_static_params

hvac_static_params = hvac_static_params()


def map_amplitude_cluster_id(epoch_df, cluster_info, logger_hvac):

    """
    Function to map amplitudes with corresponding cluster Ids

    Parameters:

        epoch_df (pd.DataFrame)         : Dataframe containing epoch level data
        cluster_info (dict)             : Dictionary containing cluster level specific key info
        logger_hvac (logging object)    : Logger to keep track of code flow

    Returns:

    """

    epoch_df['amplitude_cluster_id'] = 0
    temp_df = epoch_df.copy()

    if len(cluster_info) == hvac_static_params['length_of_cluster_info']:

        cluster_limits = cluster_info['cluster_limits']

        if len(cluster_limits) == hvac_static_params['length_of_cluster_limits']:

            temp_df['first_cluster'] = (temp_df['energy'] > cluster_limits[0][0]) & (temp_df['energy'] < cluster_limits[0][1])
            logger_hvac.info(' > Total epochs in first cluster : {} |'.format(np.sum(temp_df['first_cluster'])))

            temp_df['second_cluster'] = (temp_df['energy'] > cluster_limits[1][0]) & (temp_df['energy'] < cluster_limits[1][1])
            logger_hvac.info(' > Total epochs in second cluster : {} |'.format(np.sum(temp_df['second_cluster'])))

            temp_df['outlier'] = ~(temp_df['first_cluster'] | temp_df['second_cluster'])
            logger_hvac.info(' > Total epochs in outlier : {} |'.format(np.sum(temp_df['outlier'])))

            temp_df['amplitude_cluster_id'][temp_df['first_cluster'] == 1] = 0
            temp_df['amplitude_cluster_id'][temp_df['second_cluster'] == 1] = 1
            temp_df['amplitude_cluster_id'][temp_df['outlier'] == 1] = -1

    epoch_df['amplitude_cluster_id'] = temp_df['amplitude_cluster_id']

    return


def filter_by_setpoint(hvac_params, estimation_debug, epoch_hvac_filtered_data, mode_idx, temperature, day_idx):

    """
        Function to filtering out non-hvac data before hvac consumption estimates are made

        Parameters:
            hvac_params (dict)                    : Dictionary containing hvac algo related initialized parameters
            estimation_debug (dict)               : Dictionary containing hvac estimation stage attributes for setpoint
            epoch_hvac_filtered_data (np.ndarray) : Array containing partially filtered data
            mode_idx (int)                        : Mode identifier within loop
            temperature (np.ndarray)              : Array containing temperature info at epoch level, partial filter
            day_idx (np.ndarray)                  : Array of day identifier indexes (epochs of a day have same index)
        Returns:
            epoch_degree_arm (np.ndarray)         : Array containing epoch level degree post filtering
            degree_for_day (np.ndarray)           : Array containing day level degree post filtering
            hours_selected (np.ndarray)           : Array ccontaining info for valid epochs till this stage of filtering
            epoch_hvac_filtered_data (np.ndarray) : Array containing partially filtered data, till setpoint stage
            filter_info (dict)                    : Dictionary containing filtering related key information
        """

    if hvac_params['IS_AC']:

        min_ac_temperature_bound = estimation_debug['setpoint']
        epoch_hvac_filtered_data[np.logical_and(mode_idx, temperature <= min_ac_temperature_bound)] = 0
        epoch_hvac_filtered_data[np.logical_not(mode_idx)] = 0

        # getting daily degree_for_day aggregate for AC
        epoch_degree_arm = np.maximum(temperature - estimation_debug['setpoint'], 0)
        keep = ~np.isnan(epoch_degree_arm)
        degree_for_day = np.bincount(day_idx[keep], epoch_degree_arm[keep])
        hours_selected = epoch_degree_arm > 0

    else:

        max_sh_temperature_bound = estimation_debug['setpoint']
        epoch_hvac_filtered_data[np.logical_and(mode_idx, temperature >= max_sh_temperature_bound)] = 0
        epoch_hvac_filtered_data[np.logical_not(mode_idx)] = 0

        # getting daily degree_for_day aggregate for SH
        epoch_degree_arm = np.maximum(float(estimation_debug['setpoint']) - temperature, 0)
        keep = ~np.isnan(epoch_degree_arm)
        degree_for_day = np.bincount(day_idx[keep], epoch_degree_arm[keep])
        hours_selected = epoch_degree_arm > 0

    return epoch_degree_arm, degree_for_day, hours_selected, epoch_hvac_filtered_data


def filter_by_mode(hvac_input_data, invalid_idx, detection_debug, logger_base, estimation_debug=None,
                   day_idx=None, month_idx=None, hvac_params=None):

    """
    Function to filter out non-hvac data before hvac consumption estimates are made

    Parameters:
        hvac_input_data (numpy array)       : 2D Array of epoch level input data frame flowing into hvac module
        invalid_idx (numpy array)           : Array of invalid epochs based on consumption and temperature
        detection_debug (dict)              : Dictionary containing hvac detection stage attributes
        logger_base(logging object)         : Writes logs during code flow
        estimation_debug (dict)             : Dictionary containing hvac estimation stage attributes [for setpoint info]
        day_idx (numpy array)               : Array of day identifier indexes (epochs of a day have same index)
        month_idx (numpy array)             : Array of month identifier indexes (epochs of a month have same index)
        hvac_params (dict)                  : Dictionary containing hvac algo related initialized parameters
    Returns:
        filter_info (dict) : Dictionary containing filtering related key information
    """

    logger_local = logger_base.get("logger").getChild("filter_consumption")
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    epoch_hvac_filtered_data = copy.deepcopy(hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    epoch_hvac_filtered_data_copy = copy.deepcopy(hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    temperature_copy = copy.deepcopy(hvac_input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX])
    epoch_degree_arm = np.zeros(shape=[hvac_input_data.shape[0]])

    epoch_df = pd.DataFrame()
    epoch_df['energy'] = epoch_hvac_filtered_data_copy
    epoch_df['temperature'] = temperature_copy
    epoch_df['invalid_idx'] = invalid_idx
    epoch_df['day_idx'] = day_idx

    # Populating mode identifiers in new column of epoch_df. Name = amplitude_cluster_id
    map_amplitude_cluster_id(epoch_df, detection_debug['amplitude_cluster_info'], logger_hvac)

    unique_modes = np.unique(epoch_df['amplitude_cluster_id'])
    modes_info = detection_debug['amplitude_cluster_info']

    filter_info = {
        'degree_day' : pd.DataFrame(),
        'day_filtered_data' : pd.DataFrame(),
        'hours_selected' : pd.DataFrame(),
        'epoch_filtered_data' : pd.DataFrame(),
        'epoch_degree' : pd.DataFrame()
    }

    for mode in unique_modes:

        if estimation_debug['exist'] and not np.isnan(detection_debug['mu']) and (mode >= 0):

            mode_limits = modes_info['cluster_limits'][mode]
            mode_min_threshold = max(hvac_params['MIN_AMPLITUDE'], mode_limits[0])
            logger_hvac.info(' >> Minimum threshold for mode {} : {}W |'.format(mode, mode_min_threshold))

            epoch_hvac_filtered_data = np.array(epoch_df['energy'])
            mode_idx = np.array(epoch_df['amplitude_cluster_id'] == mode)
            mode_idx_to_suppress = np.logical_and(mode_idx, np.logical_or(invalid_idx, epoch_hvac_filtered_data < mode_min_threshold))
            logger_hvac.info(' Total epochs suppressed to zero, based on min epoch consumption : {} |'.format(np.sum(mode_idx_to_suppress)))
            epoch_hvac_filtered_data[mode_idx_to_suppress] = 0

            cap_threshold = mode_limits[1]
            mode_idx_to_cap = np.logical_and(mode_idx, epoch_hvac_filtered_data >= cap_threshold)
            logger_hvac.info(' Total epochs capped, based on cap threshold : {} |'.format(np.sum(mode_idx_to_cap)))
            epoch_hvac_filtered_data[mode_idx_to_cap] = cap_threshold

            # if epoch level consumption is zero, corresponding temperature for data filtering is set to setpoint for no regression effect.
            temperature = np.array(epoch_df['temperature'])
            temperature[np.logical_and(mode_idx, epoch_hvac_filtered_data <= 0)] = estimation_debug['setpoint']
            temperature[np.logical_not(mode_idx)] = estimation_debug['setpoint']

            epoch_degree_arm, degree_for_day, hours_selected, epoch_hvac_filtered_data = filter_by_setpoint(hvac_params, estimation_debug,
                                                                                                            epoch_hvac_filtered_data, mode_idx, temperature, day_idx)

            keep = ~np.isnan(np.logical_and(epoch_hvac_filtered_data > 0, epoch_degree_arm > 0).astype(int))
            logger_hvac.info(' Total valid epochs based on filtered consumption and degree day : {} |'.format(np.sum(keep)))

            # filtering based on minimum hours per day
            detections_daily = np.bincount(day_idx[keep], np.logical_and(epoch_hvac_filtered_data > 0, epoch_degree_arm > 0).astype(int)[keep])
            i_find = np.nonzero((detections_daily <= hvac_params['MIN_HRS_PER_DAY']).astype(int))[0]
            logger_hvac.info(' Total valid epochs suppressed to zero, based on minimum epochs per day : {} |'.format(np.sum(i_find)))
            epoch_hvac_filtered_data[np.in1d(day_idx, i_find)] = 0

            # filtering based on minimum hours per month
            detections_monthly = np.bincount(month_idx[keep], np.logical_and(epoch_hvac_filtered_data > 0, epoch_degree_arm > 0).astype(int)[keep])
            j_find = np.nonzero(detections_monthly <= hvac_params['MIN_HRS_PER_MONTH'])
            logger_hvac.info(' Total valid epochs suppressed to zero, based on minimum epochs per month : {} |'.format(np.sum(j_find)))
            epoch_hvac_filtered_data[np.in1d(month_idx, j_find)] = 0

            # getting filtered data daily consumption aggregate
            day_hvac_filtered_data = np.bincount(day_idx, epoch_hvac_filtered_data)

            filter_info['epoch_degree'][mode] = epoch_degree_arm
            filter_info['degree_day'][mode] = degree_for_day
            filter_info['hours_selected'][mode] = hours_selected
            filter_info['epoch_filtered_data'][mode] = epoch_hvac_filtered_data
            filter_info['day_filtered_data'][mode] = day_hvac_filtered_data

        elif mode >= 0:

            mode = 0

            if hvac_params['IS_AC']:

                # no setpoint exists for AC. no regression. zero vector: degree_for_day for AC
                logger_hvac.info(' >> F0 : No setpoint exists for AC. no regression |')
                logger_hvac.info(' >> Zero vector: degree_for_day for AC |')
                degree_for_day = np.zeros(shape=[len(np.unique(day_idx)), 1])
                logger_hvac.info(' >> Getting un-filtered data daily consumption aggregate for AC |')
                # getting un-filtered data daily consumption aggregate for AC
                day_hvac_filtered_data = np.bincount(day_idx, epoch_hvac_filtered_data)
                logger_hvac.info(' Getting hours selected based on temperature for AC |')
                # getting hours selected based on temperature for AC
                hours_selected = epoch_degree_arm > 0

            else:

                # no setpoint exists for SH. no regression. zero vector: degree for day  for SH
                logger_hvac.info(' >> F0 : No setpoint exists for SH. no regression |')
                logger_hvac.info(' >> Zero vector: degree for day for SH |')
                degree_for_day = np.zeros(shape=[len(np.unique(day_idx)), 1])
                logger_hvac.info(' >> Getting un-filtered data daily consumption aggregate for SH |')
                # getting un-filtered data daily consumption aggregate for SH
                day_hvac_filtered_data = np.bincount(day_idx, epoch_hvac_filtered_data)
                logger_hvac.info(' Getting hours selected based on temperature for SH |')
                # getting hours selected based on temperature for SH
                hours_selected = epoch_degree_arm > 0

            filter_info['epoch_degree'][mode] = epoch_degree_arm
            filter_info['degree_day'][mode] = np.zeros(len(np.unique(day_idx)))
            filter_info['hours_selected'][mode] = hours_selected
            filter_info['epoch_filtered_data'][mode] = epoch_hvac_filtered_data
            filter_info['day_filtered_data'][mode] = day_hvac_filtered_data

    return filter_info
