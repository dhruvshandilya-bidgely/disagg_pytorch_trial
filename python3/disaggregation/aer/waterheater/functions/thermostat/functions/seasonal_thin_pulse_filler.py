"""
Author - Sahana M
Date - 20/05/2021
This module is used to interpolate thin and fat pulses on the bill cycles with 0 estimation
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.maths_utils import convolve_function
from python3.utils.maths_utils.rolling_function import rolling_function
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.post_processing_utils import mark_fat_pulse
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.insignificant_fat_pulse_removal import remove_insignificant_fat_pulses

fat_box_columns = {
    'day_ts': 0,
    'start_idx': 1,
    'end_idx': 2,
    'valid_fat_run': 3
}


def seasonal_thin_pulse_filler(debug, fat_signal, wh_config):
    """
    This function is used to fill the months with missing wh consumption
    Parameters
    debug                   (dict)          : Dictionary containing algorithm outputs
    fat_signal              (np.ndarray)    : 21 column matrix containing fat pulse estimation
    wh_config               (dict)          : Dictionary containing water heater configurations

    Returns
    previous_thin_output    (np.ndarray)    : 21 column matrix containing interpolated wh thin pulses on missing months
    previous_fat_output     (np.ndarray)    : 21 column matrix containing interpolated wh fat pulses on missing months
    """

    # Extract the necessary variables required
    min_amp_cap = wh_config['thermostat_wh']['estimation']['min_amp_cap']
    max_amp_cap = wh_config['thermostat_wh']['estimation']['max_amp_cap']
    fat_min_cap = wh_config['thermostat_wh']['estimation']['fat_min_cap']
    fat_max_cap = wh_config['thermostat_wh']['estimation']['fat_max_cap']
    thin_pulse_amp_std = wh_config['thermostat_wh']['estimation']['thin_pulse_amp_std']

    factor = int(Cgbdisagg.SEC_IN_HOUR / wh_config['sampling_rate'])

    fat_consumption_col = 1

    # Taking a deepcopy of fat data to keep local instances

    fat_data = deepcopy(fat_signal)

    # Get unique bill cycles and its timestamps

    bill_cycle_ts, bill_cycle_idx = np.unique(fat_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_inverse=True)

    # Initialize the fractions table (fat data out of max consumption)

    bill_cycle_fractions = np.hstack([bill_cycle_ts.reshape(-1, 1), np.zeros((len(bill_cycle_ts), 2))])

    # Get fat consumption at each bill cycle

    bill_cycle_fractions[:, fat_consumption_col] = np.bincount(bill_cycle_idx,
                                                               fat_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    consumptions = bill_cycle_fractions[:, fat_consumption_col]

    # get a boolean array for zero consumption bill cycles

    zero_consumption_billcycles = consumptions == 0

    # for the zero consumption seasons perform the following interpolations

    if any(zero_consumption_billcycles):

        # Thin pulse interpolation --------------------------------------------------------------------------------

        # get the original input array after baseload removal

        wh_input = deepcopy(debug['input_data_baseload_removal'])

        # get the index of zero consumption billcycles

        missed_bill_cycles_index = np.where(zero_consumption_billcycles)[0]

        # get a boolean array of time stamps in the zero consumption bill cycles

        missed_ts_bool = np.in1d(bill_cycle_idx, missed_bill_cycles_index)

        # subset only those zero consumption time stamps

        missed_thin_input = wh_input[missed_ts_bool][:, :]

        # Get the convolution filters
        amp_mask = np.flipud([-1 / 2, 1, -1 / 2])
        der_mask = np.flipud([-1, 0, 1])

        # Energy diff to find points with consecutive increase and decrease of above min_amp

        energy_diff = np.r_[[0], np.diff(missed_thin_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]), [0] * 10]
        abs_energy_diff = np.abs(energy_diff)

        # Convolve single peak detection and difference filters

        amp = convolve_function(missed_thin_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], amp_mask)
        der = convolve_function(missed_thin_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], der_mask)

        # get the already predicted thin pulse energy for other seasons

        thin_pulse_energy = debug['thin_peak_energy']
        min_amp = min_amp_cap * thin_pulse_energy
        max_amp = max_amp_cap * thin_pulse_energy

        # Take the indices of peaks that satisfy boundary conditions

        single_peaks_bool = (amp > min_amp) & (amp < max_amp) & (der < thin_pulse_amp_std) & (abs_energy_diff[:-10] > min_amp)

        # mask the index where single_peaks_bool is True as 1 else 0

        thin_pulses_peaks = np.where(single_peaks_bool == True, 1, 0)

        # update the thin pulse consumption

        missed_thin_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = missed_thin_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] * thin_pulses_peaks

        # Removing too frequent thin pulses (minimum gap between thin pulses is 4 hours)

        thin_energy_idx = (missed_thin_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0).astype(int)
        thin_idx_diff = np.diff(np.r_[0, thin_energy_idx, 0])

        # get the start index of all the detected thin pulses

        thin_start_idx = np.where(thin_idx_diff[:-1] > 0)[0]

        valid_thin_idx = [1] * len(thin_start_idx)
        for index in range(len(thin_start_idx) - 1):

            # if the duration between the start of 2 thin pulses in < 1 hour then remove that pulse

            if (thin_start_idx[index + 1] - thin_start_idx[index]) < 1 * factor:
                valid_thin_idx[index + 1] = 0

        # get the valid thin pulse start indexes

        thin_start_idx = thin_start_idx[(thin_start_idx * valid_thin_idx) > 0]

        # mark all the invalid thin consumption as 0

        valid_thin_idx = [0]*len(missed_thin_input)
        for index in thin_start_idx :
            valid_thin_idx[index] = 1

        # Mark all the invalid consumptions as 0 and the value as its energy value

        missed_thin_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = missed_thin_input[:,Cgbdisagg.INPUT_CONSUMPTION_IDX] * valid_thin_idx

        # Update the thin pulse consumption in the final_thin_output array

        previous_thin_output = deepcopy(debug['final_thin_output'])
        previous_thin_output[missed_ts_bool, Cgbdisagg.INPUT_CONSUMPTION_IDX] = missed_thin_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

        # subtract thin pulse consumption from original wh input

        initial_wh_input = deepcopy(debug['input_data_baseload_removal'])
        initial_wh_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = initial_wh_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] - \
                                                               previous_thin_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

        # Fat pulse interpolation --------------------------------------------------------------------------------

        fat_input = deepcopy(initial_wh_input)

        # get the time stamps of zero consumption bill cycles

        missed_fat_input = fat_input[missed_ts_bool][:, :]

        # get the previously estimated fat pulse amplitude

        fat_amp = debug['new_fat_amp']

        # Fat pulse energy values

        fat_energy_values = missed_fat_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

        # Step - 1 : Detect fat pulses satisfying a fat pulse amplitude range ----------------

        # Take the indexes of the peaks that satisfy the boundary conditions

        fat_energy_peaks = (fat_energy_values > fat_min_cap * fat_amp) & (fat_energy_values < fat_max_cap * fat_amp)

        min_window = int(wh_config['thermostat_wh']['detection']['minimum_box_size'] / wh_config['sampling_rate'])

        max_duration = wh_config['thermostat_wh']['estimation']['max_fat_pulse_duration']

        # Making sure that the window is at least of size 1 unit

        min_window = np.fmax(min_window, 1)

        # Calculate the half window size (in terms of data-points)

        window_half = int(min_window // 2)

        # Define the max window size based on max duration

        max_window = int(max_duration * (Cgbdisagg.SEC_IN_HOUR / wh_config.get('sampling_rate')))

        # Accumulate the valid count over the window

        mov_sum = rolling_function(fat_energy_peaks, min_window, 'sum')

        # Filter the chunks of length in the defined window range

        valid_sum_bool = (mov_sum >= min_window) & (mov_sum <= max_window)

        sum_idx = np.where(valid_sum_bool)[0]
        sum_final = deepcopy(sum_idx)

        # Padding the boxes for the first and last window

        for i in range(1, window_half + 1):
            sum_final = np.r_[sum_final, sum_idx + i]

            if i != window_half:
                sum_final = np.r_[sum_final, sum_idx - i]

        # Updating the valid sum bool

        valid_sum_bool[sum_final[sum_final < missed_fat_input.shape[0]]] = True

        valid_fat_idx = np.where(valid_sum_bool == True, 1, 0)

        # Make all invalid consumption points as zero in fat pulse data

        missed_fat_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] *=  valid_fat_idx

        # Step -2 : Remove pulses occurring in insignificant hours ----------------

        # Get the time of usage histogram of fat pulse consumption for the rest of the seasons throughout the year.

        total_fat_input_yearly = deepcopy(fat_signal)
        fat_energy_idx_yearly = (total_fat_input_yearly[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0).astype(int)
        fat_idx_diff_yearly = np.diff(np.r_[0, fat_energy_idx_yearly, 0])
        wh_start_idx_yearly = np.where(fat_idx_diff_yearly[:-1] > 0)[0]

        # get the start and end index of the detected fat pulse boxes

        max_hod = int(1 * Cgbdisagg.HRS_IN_DAY) - 1
        edges = np.arange(0, max_hod + 2) - 0.5
        start_hod_count_yearly, _ = np.histogram(total_fat_input_yearly[wh_start_idx_yearly, Cgbdisagg.INPUT_HOD_IDX],
                                                 bins=edges)

        # Duration of the day which has max fat pulse consumption during the other seasons.

        year_round_max_hod = np.max(start_hod_count_yearly)

        # Get the time of usage histogram of fat pulse consumption for the zero consumption billcycles

        fat_energy_idx = (missed_fat_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0).astype(int)
        fat_idx_diff = np.diff(np.r_[0, fat_energy_idx, 0])

        # get the start and end index of the detected fat pulse boxes

        wh_start_idx_missed = np.where(fat_idx_diff[:-1] > 0)[0]
        wh_end_idx_missed = np.where(fat_idx_diff[1:] < 0)[0]

        # stack the start and end index into usages array

        usages = np.vstack(
            (missed_fat_input[wh_start_idx_missed, Cgbdisagg.INPUT_DAY_IDX], wh_start_idx_missed, wh_end_idx_missed)).T

        unique_days, count = np.unique(usages[:, fat_box_columns['day_ts']], return_counts=True)
        unique_days = np.vstack((unique_days, count))

        max_hod = int(1 * Cgbdisagg.HRS_IN_DAY) - 1

        edges = np.arange(0, max_hod + 2) - 0.5

        # get the histogram of hod usage of fat pulses detected for the zero consumption billcycles

        start_hod_count, _ = np.histogram(missed_fat_input[wh_start_idx_missed, Cgbdisagg.INPUT_HOD_IDX], bins=edges)

        # remove fat pulses occurring in insignificant hours

        valid_fat_run = remove_insignificant_fat_pulses(start_hod_count, start_hod_count_yearly, year_round_max_hod,
                                                        usages, unique_days, missed_fat_input, wh_config)

        # stack the valid_fat_run array into usages array

        usages = np.hstack((usages, np.asarray(valid_fat_run).reshape(-1, 1)))

        # Get the new fat pulse energy values

        fat_energy_values = mark_fat_pulse(usages, missed_fat_input)

        # update the fat pulse consumption

        missed_fat_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = fat_energy_values

        # update the final fat output

        previous_fat_output = deepcopy(debug['final_fat_output'])
        previous_fat_output[missed_ts_bool, Cgbdisagg.INPUT_CONSUMPTION_IDX] = missed_fat_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

        # update the final wh output
        previous_wh_output = deepcopy(debug['final_wh_signal'])
        previous_wh_output[missed_ts_bool, Cgbdisagg.INPUT_CONSUMPTION_IDX] = missed_thin_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] + \
                                                missed_fat_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

        # update the final consumptions in debug object

        debug['final_thin_output'] = previous_thin_output
        debug['final_fat_output'] = previous_fat_output
        debug['final_wh_signal'] = previous_wh_output

        return previous_thin_output, previous_fat_output

    else:
        return debug['final_thin_output'], debug['final_fat_output']
