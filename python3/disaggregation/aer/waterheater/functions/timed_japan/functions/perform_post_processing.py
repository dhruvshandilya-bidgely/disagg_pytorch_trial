"""
Author - Sahana M
Date - 07/06/2021
Post processing
"""

# Import python packages
import logging
import numpy as np
from copy import deepcopy

# Import packages from within the project
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.heating_inst_post_processing import heating_inst_check
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.top_down_interpolation import top_down_interpolation
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.horizontal_interpolation import horizontal_interpolation


def repopulate(final_twh_matrix, original_data_matrix, day, start_time, end_time, timed_wh_amp, wh_config):
    """
    This function is used to do a final repopulation on missing data points
    Parameters:
        original_data_matrix             (np.ndarray)        : Contains raw data
        final_twh_matrix                 (np.ndarray)        : Final timed wh estimation
        day                              (int)               : Day under consideration
        start_time                       (int)               : Start time of the band
        end_time                         (int)               : End time of the band
        timed_wh_amp                     (np.ndarray)        : Timed wh amplitude
        wh_config                        (dict)              : Timed wh configurations
    Returns:
        final_twh_matrix                 (np.ndarray)        : Final timed wh estimation
    """

    # Extract all the necessary data

    factor = wh_config.get('factor')
    past_days = max(day - 3, 0)
    base_amp = wh_config.get('re_population_base_amp')
    future_days = min(day + 3, final_twh_matrix.shape[0])

    # Extract the final twh matrix consumption of the near by days
    if day == 0:
        total_consumption = np.sum(final_twh_matrix[day:future_days, start_time:end_time])
    elif day == final_twh_matrix.shape[0] - 1:
        total_consumption = np.sum(final_twh_matrix[past_days:day, start_time:end_time])
    else:
        total_consumption = np.sum(final_twh_matrix[past_days:day, start_time:end_time]) + np.sum(
            final_twh_matrix[day:future_days, start_time:end_time])

    # If the total consumption of nearby days > 0
    if total_consumption:

        # Get the consumption on the current day from original & final matrix

        original_day_consumption = original_data_matrix[day, start_time:end_time].copy()
        band_matrix_consumption = final_twh_matrix[day, start_time:end_time].copy()

        # Get all the indexes where the consumption lies within a range
        original_day_consumption[original_day_consumption < max(timed_wh_amp - (base_amp / factor), 0)] = 0
        original_day_consumption[original_day_consumption > (timed_wh_amp + (base_amp / factor))] = 0
        original_day_consumption_bool = original_day_consumption > 0

        # Get all the indexes where the consumption lies within a range
        band_matrix_consumption[band_matrix_consumption < max(timed_wh_amp - (base_amp / factor), 0)] = 0
        band_matrix_consumption[band_matrix_consumption > (timed_wh_amp + (base_amp / factor))] = 0
        band_matrix_consumption_bool = band_matrix_consumption > 0

        # If the current day consumption is 0, replace it with the raw data consumption after conditions match
        if np.sum(band_matrix_consumption_bool) == 0 and np.sum(original_day_consumption) > 0:
            band_matrix_consumption[original_day_consumption_bool] = original_day_consumption[original_day_consumption_bool]
            final_twh_matrix[day, start_time:end_time] = np.maximum(band_matrix_consumption, final_twh_matrix[day, start_time:end_time])

    return final_twh_matrix


def re_population(original_data_matrix, final_twh_matrix, start_time, end_time, wh_config):
    """
    This function is used to do a final repopulation on missing data points
    Parameters:
        original_data_matrix             (np.ndarray)        : Contains raw data
        final_twh_matrix                 (np.ndarray)        : Final timed wh estimation
        start_time                       (int)               : Start time of the band
        end_time                         (int)               : End time of the band
        wh_config                        (dict)              : Timed wh configurations
    Returns:
        final_twh_matrix                 (np.ndarray)        : Final timed wh estimation
    """

    # Identify the final timed wh amp & other time info
    window = wh_config.get('factor')
    start_time = int(max(start_time - window, 0))
    end_time = int(min(end_time + window, final_twh_matrix.shape[1]))
    timed_wh_amp = np.median(final_twh_matrix[final_twh_matrix > 0])

    # Do top down re_population
    for day in range(original_data_matrix.shape[0]):
        final_twh_matrix = repopulate(final_twh_matrix, original_data_matrix, day, start_time, end_time, timed_wh_amp, wh_config)

    # Fo down top re_population
    for day in range(original_data_matrix.shape[0]-1, -1, -1):
        final_twh_matrix = repopulate(final_twh_matrix, original_data_matrix, day, start_time, end_time, timed_wh_amp, wh_config)

    return final_twh_matrix


def get_day_wise_amp(season_usage, vacation_day_bool):
    """This function takes in every day consumption data & calculates the median amplitude by excluding vacation days"""
    day_wise_median = []
    for i in range(season_usage.shape[0]):
        temp = season_usage[i, season_usage[i, :] > 0]

        # Exclude vacation days & non zero days
        if not vacation_day_bool[i] and len(temp):
            day_wise_median.append(np.median(temp))

    return day_wise_median


def amp_capping(final_twh_matrix, start_time, end_time, wh_config, debug):
    """
    This function is used to perform variable amplitude capping (comparative capping between winter & summer)
    Parameters:
        final_twh_matrix                 (np.ndarray)        : Final timed wh estimation
        start_time                       (int)               : Start time of the band
        end_time                         (int)               : End time of the band
        wh_config                        (dict)              : Timed wh configurations
        debug                            (dict)              : Contains algorithm outputs
    Returns:
        final_twh_matrix                 (np.ndarray)        : Final timed wh estimation
    """

    # Extract all the necessary data
    min_amp = wh_config.get('min_amp')
    factor = wh_config.get('factor')
    num_hours = factor * 2
    end_time = int(min((end_time + num_hours), final_twh_matrix.shape[1]))
    start_time = int(max(0, (start_time - num_hours)))

    s_label = debug.get('season_label')
    vacation_day_bool = debug.get('vacation_days_bool')

    # Perform variable capping only if winter is available
    perform_variable_capping = (len(np.unique(s_label)) > 1) & (np.isin(-1, s_label))

    if perform_variable_capping:

        # Get all the winter days and identify their usage
        winter_days_bool = (s_label == -1)
        winter_usage = final_twh_matrix[winter_days_bool, start_time:end_time]
        other_season_usage = final_twh_matrix[~winter_days_bool, start_time:end_time]

        # Get the amplitude median of winter & other seasons excluding vacation days
        winter_daywise_median = get_day_wise_amp(winter_usage, vacation_day_bool)
        other_daywise_median = get_day_wise_amp(other_season_usage, vacation_day_bool)
        winter_amp_median = np.median(winter_daywise_median)
        other_amp_median = np.median(other_daywise_median)

        # identify the capping amplitude based on the difference ratio
        if winter_amp_median >= other_amp_median:
            ratio = np.round(other_amp_median/winter_amp_median, 2)
            amplitude_buffer = (1 - ratio) * 0.98 * winter_amp_median
            amp_cap = other_amp_median + amplitude_buffer + int((min_amp/2)/factor)
        else:
            ratio = np.round(winter_amp_median/other_amp_median, 2)
            amplitude_buffer = (1 - ratio) * 0.98 * other_amp_median
            amp_cap = winter_amp_median + amplitude_buffer + int(min_amp/factor)

        # Cap the outlier amplitudes
        temp = final_twh_matrix[:, start_time:end_time]
        temp[temp >= amp_cap] = amp_cap

    # If no winter available for comparision then perform general capping
    else:
        # Identify the year round usage
        year_round_usage = final_twh_matrix[:, start_time:end_time]

        # Get the amplitude median of the whole year excluding vacation days
        year_daywise_median = get_day_wise_amp(year_round_usage, vacation_day_bool)

        # Get the final capping amplitude
        twh_amp = np.median(year_daywise_median)
        amp_cap = int(min_amp/factor) + twh_amp

        # Cap the outlier amplitudes
        temp = final_twh_matrix[:, start_time:end_time]
        temp[temp >= amp_cap] = amp_cap

    # Cap the outlier amplitudes in the lower end
    low_amp_bar = np.percentile(temp[temp > 0], q=5)
    temp[temp < low_amp_bar] = 0

    return final_twh_matrix


def post_processing(debug, wh_config, logger_base):
    """
    Perform post processing
    Parameters:
        debug                   (dict)              : Contains algorithm output
        wh_config               (dict)              : WH configurations dictionary
        logger_base             (logger)            : Logger passed

    Returns:

    """

    # Initialise logger for this module

    logger_local = logger_base.get('logger').getChild('post_processing')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    logger.info('Starting Post processing for the Timed WH | ')

    # Extract the necessary data

    final_twh_matrix = debug.get('final_twh_matrix')
    overall_chunk_data = debug.get('overall_chunk_data')
    original_data_matrix = deepcopy(debug.get('original_data_matrix'))

    twh_bands = debug.get('twh_bands')
    logger.info('Total number of timed wh bands to post process | {} '.format(len(twh_bands)))

    for band in twh_bands:
        start_time = debug['bands_info']['band_' + str(band)]['start_time_' + str(band)]
        end_time = debug['bands_info']['band_' + str(band)]['end_time_' + str(band)]

        # Check for heating instance

        final_twh_matrix = heating_inst_check(original_data_matrix, final_twh_matrix, start_time, end_time, wh_config, debug)

        # Top down interpolation

        final_twh_matrix = top_down_interpolation(original_data_matrix, final_twh_matrix, start_time, end_time, wh_config, overall_chunk_data)

        # Horizontal interpolation

        final_twh_matrix = horizontal_interpolation(original_data_matrix, final_twh_matrix, start_time, end_time, wh_config, overall_chunk_data)

        # Final re-population of missed data points

        final_twh_matrix = re_population(original_data_matrix, final_twh_matrix, start_time, end_time, wh_config)

        # Amplitude capping

        final_twh_matrix = amp_capping(final_twh_matrix, start_time, end_time, wh_config, debug)

    # Avoiding the residual timed pattern

    final_twh_matrix = removed_timed_residual(final_twh_matrix, original_data_matrix)
    logger.info('Removed timed residual | ')

    # Identify the final TWh amplitude

    twh_boxes = final_twh_matrix[final_twh_matrix > 0]
    twh_amplitude = np.nanmedian(twh_boxes)
    debug['timed_wh_amplitude'] = twh_amplitude * wh_config.get('factor')

    return final_twh_matrix, debug


def removed_timed_residual(final_twh_matrix, original_data_matrix):
    """
    Remove an timed residual left at the sides of timed wh
    Parameters:
        final_twh_matrix        (np.ndarray)    : Final timed wh matrix
        original_data_matrix    (np.ndarray)    : Original data matrix
    Returns:
        final_twh_matrix        (np.ndarray)    : Final timed wh matrix
    """

    # Extract the required variables

    twh_bool = final_twh_matrix > 0
    zero_array = np.full(shape=(twh_bool.shape[0], 1), fill_value=0)
    box_energy_idx_diff = np.diff(np.c_[zero_array,  twh_bool.astype(int), zero_array])
    corners_bool = box_energy_idx_diff[:, :-1]

    # For each day remove any timed residual

    for i in range(twh_bool.shape[0]):
        for j in range(twh_bool.shape[1]):

            # At the start of the twh if the amplitude is within the range include it in final estimation

            if corners_bool[i][j] == 1 and j > 0 and (original_data_matrix[i][j-1] > 0.20*final_twh_matrix[i, j]) and \
                    (original_data_matrix[i][j-1] < 2*final_twh_matrix[i, j]):
                final_twh_matrix[i][j-1] = original_data_matrix[i][j-1]

            # At the end of the twh if the amplitude is within the range include it in final estimation

            if corners_bool[i][j] == -1 and j <= twh_bool.shape[0]-2 and (original_data_matrix[i][j] > 0.20*final_twh_matrix[i, j-1]) and \
                    (original_data_matrix[i][j] < 2*final_twh_matrix[i, j-1]):
                final_twh_matrix[i][j] = original_data_matrix[i][j]

    return final_twh_matrix
