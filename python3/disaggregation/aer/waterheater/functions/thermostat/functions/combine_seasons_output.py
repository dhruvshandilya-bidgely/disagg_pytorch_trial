"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to combine the thin and fat pulse usages of all seasons together
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.post_processing_utils import mark_fat_pulse2

fat_box_columns = {
    'day_ts': 0,
    'start_idx': 1,
    'end_idx': 2,
    'duration': 3,
    'close_thin_pulse': 4,
    'valid_fat_run': 5
}


def combining_seasons_output(debug, logger_base):
    """
    Parameters:
        debug               (dict)      : Algorithm intermediate steps output
        logger_base         (dict)      : Logger object

    Returns:
        debug               (dict)      : Algorithm intermediate steps output
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('combining_seasons_output')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Extract all season feature dictionaries

    features = debug['season_features']

    # Combine fat box output

    total_box_output = np.vstack((features['wtr']['box_output'],
                                  features['itr']['box_output'],
                                  features['smr']['box_output']))

    # Combine thin pulse output

    total_thin_output = np.vstack((features['wtr']['thin_output'],
                                   features['itr']['thin_output'],
                                   features['smr']['thin_output']))

    # Combine fat pulse output

    total_fat_output = np.vstack((features['wtr']['fat_output'],
                                  features['itr']['fat_output'],
                                  features['smr']['fat_output']))

    # Sort all output with respect to epoch timestamp column

    total_box_output = total_box_output[total_box_output[:, Cgbdisagg.INPUT_EPOCH_IDX].argsort()]
    total_thin_output = total_thin_output[total_thin_output[:, Cgbdisagg.INPUT_EPOCH_IDX].argsort()]
    total_fat_output = total_fat_output[total_fat_output[:, Cgbdisagg.INPUT_EPOCH_IDX].argsort()]

    # Subset non zero fat energy values

    fat_non_zero = total_fat_output[total_fat_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0,
                                    Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Plot fat energy values histogram and find fat amplitude

    fat_energy_count, fat_energy_edges = np.histogram(fat_non_zero)
    fat_energy_edges = (fat_energy_edges[1:] + fat_energy_edges[:-1]) / 2

    debug['new_fat_amp'] = fat_energy_edges[np.argmax(fat_energy_count)]

    # Check if non-zero fat consumption found

    if len(fat_non_zero) > 0:
        # Valid fat energy found, find lower and upper bounds

        debug['new_fat_lamp'] = np.min(fat_non_zero)
        debug['new_fat_uamp'] = np.max(fat_non_zero)
    else:
        # No fat usage found, make bounds zero

        debug['new_fat_lamp'] = 0
        debug['new_fat_uamp'] = 0

        logger.info('No fat consumption found | ')

    # Store final consumption output to debug object

    debug['final_box_output'] = total_box_output[:, :Cgbdisagg.INPUT_DIMENSION]
    debug['final_thin_output'] = total_thin_output[:, :Cgbdisagg.INPUT_DIMENSION]
    debug['final_fat_output'] = total_fat_output[:, :Cgbdisagg.INPUT_DIMENSION]

    # Add thin and fat pulse output to get total water heater consumption

    final_output = deepcopy(total_thin_output)
    final_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] += total_fat_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Save final water heater signal to debug

    debug['final_wh_signal'] = deepcopy(final_output)

    # Calculate the residual consumption (raw consumption - water heater consumption)

    residual = deepcopy(debug['input_data'])
    residual[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] -= final_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    debug['residual'] = residual

    # Aggregate user level thin and fat consumption

    debug['new_thin_consumption'] = np.sum(total_thin_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    debug['new_fat_consumption'] = np.sum(total_fat_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # Aggregate the maximum allowed thin consumption

    debug['max_thin_output'] = features['wtr']['max_thin_output'] + \
                               features['itr']['max_thin_output'] + \
                               features['smr']['max_thin_output']

    return debug


def fat_noise_removal(total_fat_output, total_thin_output, wh_config):
    """
    Parameters:
        total_fat_output          (np.ndarray)      : 21 column matrix containing the fat pulses estimation
        total_thin_output         (np.ndarray)      : 21 column matrix containing the fat pulses estimation
        wh_config                 (dict)            : Contains all the WH configurations

    Returns:
        total_fat_output          (np.ndarray)      : 21 column matrix containing the fat pulses estimation
    """

    # Extract the necessary data
    sampling_rate = wh_config['sampling_rate']
    max_fat_runs = wh_config['thermostat_wh']['estimation']['max_fat_runs']
    fat_noise_hod_thr = wh_config['thermostat_wh']['estimation']['fat_noise_hod_thr']
    fat_noise_max_hours = wh_config['thermostat_wh']['estimation']['fat_noise_max_hours']

    # Get the detected fat indices

    fat_energy_idx = (total_fat_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0).astype(int)
    fat_idx_diff = np.diff(np.r_[0, fat_energy_idx, 0])

    # get the start and end index of the detected fat pulse boxes

    wh_start_idx = np.where(fat_idx_diff[:-1] > 0)[0]
    wh_end_idx = np.where(fat_idx_diff[1:] < 0)[0]

    # stack the start and end index into usages array

    usages = np.vstack((total_fat_output[wh_start_idx, Cgbdisagg.INPUT_DAY_IDX], wh_start_idx, wh_end_idx)).T

    # calculate and stack the duration of each fat pulse

    boxes_duration = ((usages[:, fat_box_columns['end_idx']] -
                       usages[:, fat_box_columns['start_idx']]) + 1).reshape(-1, 1)

    usages = np.hstack((usages, boxes_duration))

    # Calculate the energy consumption and closest thin pulse existence (<1 hour) for each fat pulse

    factor = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    # Get the unique days with fat pulses

    unique_days, count = np.unique(usages[:, 0], return_counts=True)
    unique_days = np.vstack((unique_days, count))

    max_hod = int(1 * Cgbdisagg.HRS_IN_DAY) - 1

    edges = np.arange(0, max_hod + 2) - 0.5

    # Get the distribution of fat pulse w.r.t hour of day

    start_hod_count, _ = np.histogram(total_fat_output[wh_start_idx, Cgbdisagg.INPUT_HOD_IDX], bins=edges)

    valid_fat_run = []
    close_thin_pulse = []
    for index in range(len(usages)):

        # get the end idx, no_of_runs_index, no_of_runs
        end_idx = int(usages[index, fat_box_columns['end_idx']])
        no_of_runs_index = np.where(unique_days[0, :] == usages[index, 0])
        no_of_runs = unique_days[1, no_of_runs_index]

        # mark all the indexes as 1 if a close thin peak is found (thin peak within an hour of fat pulse)
        if np.sum(total_thin_output[end_idx:end_idx + factor, Cgbdisagg.INPUT_CONSUMPTION_IDX]) > 0:
            close_thin_pulse.append(1)
        else:
            close_thin_pulse.append(0)

        # if the daily number of runs > 3, then remove the pulse in an insignificant hour

        if no_of_runs >= max_fat_runs:
            hod = int(total_fat_output[int(usages[index, 1]), Cgbdisagg.INPUT_HOD_IDX])

            if start_hod_count[hod] > fat_noise_hod_thr * np.max(start_hod_count):
                valid_fat_run.append(1)
            else:
                valid_fat_run.append(0)
        else:
            valid_fat_run.append(1)

        # long duration fat pulse clipping - if fat pulse > 4 hour run then clip its ends.

        if usages[index, 3] >= factor * fat_noise_max_hours:
            usages[index, 2] = usages[index, 2] - factor

    # stack the run close thin peak existence of and valid fat for each fat pulse into usages array

    usages = np.hstack((usages, np.asarray(close_thin_pulse).reshape(-1, 1)))
    usages = np.hstack((usages, np.asarray(valid_fat_run).reshape(-1, 1)))

    # final fat output array

    fat_output = mark_fat_pulse2(usages, total_fat_output)

    # update the final fat consumption with the updated fat consumptions.

    total_fat_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = fat_output

    return total_fat_output
