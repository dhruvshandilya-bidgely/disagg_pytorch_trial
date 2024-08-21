"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
This module estimates the fat pulse consumption of water heater
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.find_peak_hours import get_peak_range
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.find_peak_hours import find_peak_hours

from python3.disaggregation.aer.waterheater.functions.thermostat.functions.fat_pulse_filter import fat_box_columns
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.fat_pulse_filter import fat_pulse_filter

from python3.disaggregation.aer.waterheater.functions.thermostat.functions.validate_fat_hours import check_fat_hours


def fat_pulse_consumption(input_data, box_input, wh_config, fat_amp, upper_fat_amp, debug, logger_base):
    """
    Parameters:
        input_data              (np.ndarray)        : Input 21-column matrix
        box_input               (np.ndarray)        : Potential fat consumption
        wh_config               (dict)              : Config params
        fat_amp                 (float)             : Optimal fat energy value
        upper_fat_amp           (float)             : Upper fat energy value
        debug                   (dict)              : Algorithm intermediate steps output
        logger_base             (dict)              : Logger object

    Returns:
        final_fat_hours         (np.ndarray)        : Final allowed fat pulse hours
        fat_consumption         (np.ndarray)        : Fat pulse consumption
        final_usages            (np.ndarray)        : Fat boxes
        debug                   (dict)              : Algorithm intermediate steps output
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('fat_pulse_consumption')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Check if input data valid

    if input_data.shape[0] == 0:
        # If input data is not available, return blank

        return [0], np.array([], dtype=np.int64).reshape(0, Cgbdisagg.INPUT_DIMENSION + 1),\
               np.array([], dtype=np.int64).reshape(0, 7), debug

    # #------------------------------- Step-1: Finding the peak fat usage hours ---------------------------------------#

    # Taking a deepcopy of input data to keep local instances and initiating fat consumption zero

    fat_consumption = deepcopy(input_data)
    fat_consumption[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    # Taking a deepcopy of box data to keep local instances

    box_data = deepcopy(box_input)
    box_energy = deepcopy(box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # Get sampling rate, max fat pulse duration and min peak distance

    sampling_rate = wh_config['sampling_rate']
    min_peak_distance = wh_config['thermostat_wh']['estimation']['min_peak_distance']
    fat_duration_limit = wh_config['thermostat_wh']['estimation']['fat_duration_limit']

    max_fat_pulse_duration = wh_config['thermostat_wh']['estimation']['max_fat_pulse_duration']

    # Finding the start and end indices of all boxes

    box_energy_idx = (box_energy > 0).astype(int)
    box_idx_diff = np.diff(np.r_[0, box_energy_idx, 0])

    wh_start_idx = np.where(box_idx_diff[:-1] > 0)[0]
    wh_end_idx = np.where(box_idx_diff[1:] < 0)[0]

    # Extract the hour of start of all boxes

    fat_hours = fat_consumption[wh_start_idx, Cgbdisagg.INPUT_HOD_IDX]

    # Aggregate the fat consumption across hours of the day

    bins = np.arange(0, Cgbdisagg.HRS_IN_DAY + 1)
    hourly_count, edges = np.histogram(fat_hours, bins=bins)

    edges = edges[:-1]

    # Find the hours with high fractions as compared to others

    peaks_size, peaks_hours, widths, proms = find_peak_hours(hourly_count, min_peak_distance, wh_config, logger)

    # #----------------------------- Step-2: Finding the adjacent fat usage hours -------------------------------------#

    # Get the peak adjacent height limit and default fat pulse hour if no peaks detected

    peak_height = wh_config['thermostat_wh']['estimation']['peak_height']
    default_hour = wh_config['thermostat_wh']['estimation']['default_hour']

    count_peaks = {'one_peak': 1, 'two_peak': 2}

    # Check if fat hours already available (incremental / mtd run mode)

    if np.any(debug.get('possible_fat_hours')) is not None:
        # If fat hours available

        final_fat_hours = debug.get('possible_fat_hours')

        logger.info('The possible fat hours found in debug | {}'.format(final_fat_hours))

    elif len(peaks_hours) == count_peaks['two_peak']:
        # If two peaks in data, final valid hours initialized

        final_fat_hours = np.array([])

        # # Operating on first peak

        # Get the left and right edges of the peak

        left_edge, right_edge = get_peak_range(edges, peaks_hours[0], peaks_size[0], hourly_count, peak_height,
                                               fat_duration_limit)

        # Add the hours between left and right edge to final fat hours

        final_fat_hours = np.r_[final_fat_hours, np.arange(left_edge, right_edge + 1)]

        # # Operating on second peak

        # Get the left and right edges of the peak

        left_edge, right_edge = get_peak_range(edges, peaks_hours[1], peaks_size[1], hourly_count, peak_height,
                                               fat_duration_limit)

        # Add the hours between left and right edge to final fat hours

        final_fat_hours = np.r_[final_fat_hours, np.arange(left_edge, right_edge + 1)]

        debug['possible_fat_hours'] = final_fat_hours

    elif len(peaks_hours) == count_peaks['one_peak']:
        # # If one peak in data

        # Get the left and right edges of the peak

        left_edge, right_edge = get_peak_range(edges, peaks_hours[0], peaks_size[0], hourly_count, peak_height,
                                               fat_duration_limit)

        # Add the hours between left and right edge to final fat hours

        final_fat_hours = np.arange(left_edge, right_edge + 1)

        debug['possible_fat_hours'] = final_fat_hours
    else:
        # If zero peaks detected, add default hour to the fat pulse hours

        final_fat_hours = np.array([default_hour])

    # Check if australia pilot

    if wh_config['pilot_id'] in PilotConstants.NEW_ZEALAND_PILOTS:
        # Get night hours bound from config

        night_hours = wh_config['thermostat_wh']['estimation']['night_hours']

        # Extract night start and end hour

        night_start = night_hours[0]
        night_end = night_hours[-1]

        final_fat_hours = np.arange(night_end, night_start + 1)

    # Validate the fat hours

    final_fat_hours = check_fat_hours(final_fat_hours, wh_config, logger)

    logger.info('The possible fat hours calculated are | {}'.format(final_fat_hours))

    # #------------------------------- Step-3: Filter the invalid fat usage boxes -------------------------------------#

    # Subset the fat boxes data overlapping with final fat hours

    fat_valid_idx = np.isin(fat_hours, final_fat_hours)

    # Get the start and end indices of valid boxes

    wh_start_idx = wh_start_idx[fat_valid_idx]
    wh_end_idx = wh_end_idx[fat_valid_idx]

    # Stack the box features containing start, end, energy etc

    usages = np.vstack((fat_consumption[wh_start_idx, Cgbdisagg.INPUT_DAY_IDX], wh_start_idx, wh_end_idx)).T

    # Calculate the run duration of each box

    boxes_duration = ((usages[:, fat_box_columns['end_idx']] -
                       usages[:, fat_box_columns['start_idx']]) + 1).reshape(-1, 1)

    usages = np.hstack((usages, boxes_duration))

    # Get the ideal and upper fat pulse energy for each box

    boxes_ideal_energy = (usages[:, fat_box_columns['duration']] * fat_amp).reshape(-1, 1)

    boxes_upper_energy = (usages[:, fat_box_columns['duration']] * upper_fat_amp).reshape(-1, 1)

    usages = np.hstack((usages, boxes_ideal_energy, boxes_upper_energy))

    # Sampling rate factor

    factor = Cgbdisagg.SEC_IN_HOUR / sampling_rate

    # Keep boxes with max energy within the upper bound and filter rest of the boxes

    usages = usages[(usages[:, fat_box_columns['duration']] <= factor * max_fat_pulse_duration)]

    # Remove the incomplete boxes occurring at the start or end of the data

    usages = usages[(usages[:, fat_box_columns['start_idx']] != 0) &
                    (usages[:, fat_box_columns['end_idx']] != (input_data.shape[0] - 1))]

    # Converting all usages to int data type

    usages = usages.astype(int)

    # Initialize array to contain box area for all boxes

    boxes_area = np.array([])

    # Iterate on each fat box

    for idx, usage in enumerate(usages):
        # Get start and end index of the current box

        start, end = usage[fat_box_columns['start_idx']], usage[fat_box_columns['end_idx']]

        # Calculate energy area of the box

        area = np.sum(input_data[start:(end + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX]) - \
               peak_height * np.sum(input_data[[start - 1, end + 1], Cgbdisagg.INPUT_CONSUMPTION_IDX])

        # Append area of current box to all boxes area array

        boxes_area = np.r_[boxes_area, area]

    # Stack the boxes area to usages array

    usages = np.hstack((usages, boxes_area.reshape(-1, 1)))

    # Filter fat usage boxes

    fat_consumption, final_usages = fat_pulse_filter(input_data, fat_consumption, usages, wh_config, logger_pass)

    return final_fat_hours, fat_consumption, final_usages, debug
