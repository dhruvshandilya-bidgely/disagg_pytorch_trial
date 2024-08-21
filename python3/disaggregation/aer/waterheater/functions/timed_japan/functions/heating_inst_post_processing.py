"""
Author - Sahana M
Date - 23/06/2021
This function is function to pefrom post processing on heating instances
"""


# Import python packages
import numpy as np


def get_amp(correct_consumption_indexes, raw_day_consumption, interpolated_twh_amp):
    """
    This function is used to get the correct amplitude
    Parameters:
        correct_consumption_indexes     (np.ndarray)    : Consumption indexes
        raw_day_consumption             (np.ndarray)    : Consumption for the day
        interpolated_twh_amp            (float)         : Interpolated timed wh amplitude
    Returns:
        correct_amp                     (float)         : Correct amplitude
    """
    correct_amp = interpolated_twh_amp
    if len(correct_consumption_indexes):
        correct_amp = np.median(raw_day_consumption[correct_consumption_indexes])
        if correct_amp == 0:
            correct_amp = interpolated_twh_amp

    return correct_amp


def replace_heating_interpolation(final_twh_matrix, original_data_matrix, wh_config, band, heating_inst_dict, band_start, band_end):
    """
    Replacing heating instance with optimal raw data
    Parameters:
        final_twh_matrix                (np.ndarray)    : 2D matrix containing the box fit data
        original_data_matrix            (np.ndarray)    : 2D matrix containing the original data
        wh_config                       (dict)          : All timed wh configurations
        band                            (int)           : The band number
        heating_inst_dict               (dict)          : Contains all info about heating instances
        band_start                      (int)           : Band start time
        band_end                        (int)           : Band end time

    Returns:
        final_twh_matrix                (np.ndarray)    : 2D matrix containing the box fit data
    """

    factor = wh_config.get('factor')
    min_amp = wh_config.get('auc_min_amp_criteria')
    heating_days_bool = heating_inst_dict['band_idx_' + str(band)]['twh_heating_days']
    heating_inst_amp = heating_inst_dict['band_idx_' + str(band)]['heating_inst_amp']

    heating_days_idx = np.where(heating_days_bool)[0]

    # For each heating presence day

    for day in heating_days_idx:

        # Identify the interpolated amplitude
        interpolated_twh_amp = np.percentile(final_twh_matrix[day, band_start:band_end], q=95)

        if interpolated_twh_amp > 0:

            # Get the raw consumption in the same time interval
            raw_day_consumption = original_data_matrix[day, band_start:band_end].copy()

            # check if the raw median consumption > interpolated amp, if so, replace the high amplitudes with interpolated amp

            if np.median(raw_day_consumption) > (interpolated_twh_amp + (min_amp / factor)):
                diff = raw_day_consumption - heating_inst_amp
                diff[diff < 0] = 0
                diff += interpolated_twh_amp
                raw_day_consumption = np.minimum(diff, raw_day_consumption)

            else:

                # Identify the abnormally high consumption indexes
                over_consumption_indexes = np.where(raw_day_consumption > (interpolated_twh_amp + (min_amp / factor)))[0]
                correct_consumption_indexes = np.where(raw_day_consumption <= (interpolated_twh_amp + (min_amp / factor)))[0]

                # If there are high consumption points, then identify the correct amplitude
                if len(over_consumption_indexes):
                    correct_amp = get_amp(correct_consumption_indexes, raw_day_consumption, interpolated_twh_amp)

                    # Replace the high consumption points with the correct amplitude
                    temp = raw_day_consumption[over_consumption_indexes]
                    heating_instance_idx = temp > interpolated_twh_amp
                    twh_margin = temp[heating_instance_idx] - heating_inst_amp
                    temp[heating_instance_idx] = twh_margin
                    temp[temp > interpolated_twh_amp] += correct_amp
                    temp[temp < interpolated_twh_amp] = 0
                    raw_day_consumption[over_consumption_indexes] = temp

            # Replace the final twh matrix with the processed amplitudes
            final_twh_matrix[day, band_start:band_end] = raw_day_consumption

    return final_twh_matrix


def heating_inst_check(original_data_matrix, final_twh_matrix, start_time, end_time, wh_config, debug):
    """
    This function is used to replace any heating instance interpolation with the optimal raw data value
    Parameters:
        original_data_matrix            (np.ndarray)    : 2D matrix containing the original data
        final_twh_matrix                (np.ndarray)    : 2D matrix containing the box fit data
        start_time                      (int)           : Start time of the band
        end_time                        (int)           : End time of the band
        wh_config                       (dict)          : All timed wh configurations
        debug                           (dict)          : Contains algorithm output
    Returns:
        final_twh_matrix                (np.ndarray)    : 2D matrix containing the box fit data
    """

    heating_inst_dict = debug.get('heating_instance_bands')

    # check if the user has heating instance and in which band

    if heating_inst_dict.get('user_heating_inst'):
        num_bands = heating_inst_dict['num_bands']

        for band in range(1, (num_bands+1)):

            if heating_inst_dict['band_idx_' + str(band)]['has_heating_instance']:
                time_coverage_bool = heating_inst_dict['band_idx_' + str(band)]['expansion_arr']

                band_start = heating_inst_dict['band_idx_' + str(band)]['start_time']
                band_end = heating_inst_dict['band_idx_' + str(band)]['end_time']

                # check if there was any interpolation
                if np.sum(time_coverage_bool) >= 1 and (band_start >= start_time) and (band_end <= end_time):

                    # Replace the interpolation with the actual raw data
                    final_twh_matrix = replace_heating_interpolation(final_twh_matrix, original_data_matrix, wh_config,
                                                                     band, heating_inst_dict, band_start, band_end)

    final_twh_matrix[final_twh_matrix < 0] = 0

    return final_twh_matrix
