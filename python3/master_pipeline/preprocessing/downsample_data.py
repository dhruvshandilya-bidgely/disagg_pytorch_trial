"""
Author - Mayank Sharan
Date - 28/09/18
Down samples data to a target GB sampling rate, takes approach based on if data is HAN or GB. Assumes continuous data
"""

# Import python packages

import numpy as np
from scipy.stats import mode

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.maths_utils.find_seq import find_seq


def group_max(bins, weight):

    """
    Parameters:
        bins                (np.ndarray)        : Array containing reverse indices based on unique taken
        weight              (np.ndarray)        : Column value at corresponding positions to be collate to one

    Returns:
        out                 (np.ndarray)        : Array containing collated values taken using max
    """

    num_bins = np.max(bins) + 1

    out = np.zeros(num_bins)
    np.maximum.at(out, bins, weight)
    return out


def downsample_data(input_data, target_rate):

    """
    Parameters:
        input_data          (np.ndarray)        : 21 column input data
        target_rate         (int)               : The sampling rate in seconds that the data has to be down sampled to

    Returns:
        downsampled_data    (np.ndarray)        : 21 column array with down sampled data
    """

    difference = np.diff(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX])

    if len(difference):
        sampling_rate = int(mode(difference)[0][0])

        # Check a few conditions on the sampling rates to determine feasibility of downsampling

        if sampling_rate == target_rate:
            downsampled_data = input_data

        elif sampling_rate > target_rate or not(target_rate % sampling_rate == 0):
            print('Can\'t handle this sampling rate')
            downsampled_data = input_data

        else:

            # Get indices and downsampled timestamps ready

            group_column = (np.ceil((input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] / target_rate)) * target_rate)
            new_epoch_ts, bin_idx = np.unique(group_column, return_inverse=True)

            # Calculate the last index in each unique seq to use for downsampling columns.
            # To align with end aggregated downsampling

            seq_end_col = 2

            seq_arr = find_seq(group_column, min_seq_length=0)
            new_epoch_idx = seq_arr[:, seq_end_col].astype(int)

            # Fill values in the downsampled array

            bin_wise_count = np.bincount(bin_idx)

            downsampled_data = np.full(shape=(len(new_epoch_ts), Cgbdisagg.INPUT_DIMENSION), fill_value=np.nan)

            downsampled_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX: Cgbdisagg.INPUT_HOD_IDX + 1] = \
                input_data[new_epoch_idx, Cgbdisagg.INPUT_BILL_CYCLE_IDX: Cgbdisagg.INPUT_HOD_IDX + 1]

            downsampled_data[:, Cgbdisagg.INPUT_EPOCH_IDX] = new_epoch_ts
            downsampled_data[:, Cgbdisagg.INPUT_S_LABEL_IDX] = input_data[new_epoch_idx, Cgbdisagg.INPUT_S_LABEL_IDX]

            if sampling_rate > 60:
                # GB to GB downsampling

                downsampled_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = \
                    np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
            else:
                # HAN to GB downsampling

                scaling_factor = float(target_rate) / Cgbdisagg.SEC_IN_HOUR

                downsampled_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = \
                    np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]) * scaling_factor / bin_wise_count

            downsampled_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]) / bin_wise_count
            downsampled_data[:, Cgbdisagg.INPUT_SKYCOV_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_SKYCOV_IDX]) / bin_wise_count
            downsampled_data[:, Cgbdisagg.INPUT_WIND_SPD_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_WIND_SPD_IDX]) / bin_wise_count
            downsampled_data[:, Cgbdisagg.INPUT_DEW_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_DEW_IDX]) / bin_wise_count

            downsampled_data[:, Cgbdisagg.INPUT_SUNRISE_IDX] = \
                group_max(bin_idx, input_data[:, Cgbdisagg.INPUT_SUNRISE_IDX])
            downsampled_data[:, Cgbdisagg.INPUT_SUNSET_IDX] = \
                group_max(bin_idx, input_data[:, Cgbdisagg.INPUT_SUNSET_IDX])

            downsampled_data[:, Cgbdisagg.INPUT_FEELS_LIKE_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_FEELS_LIKE_IDX]) / bin_wise_count

            downsampled_data[:, Cgbdisagg.INPUT_PREC_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_PREC_IDX])
            downsampled_data[:, Cgbdisagg.INPUT_SNOW_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_SNOW_IDX])

            downsampled_data[:, Cgbdisagg.INPUT_SL_PRESS_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_SL_PRESS_IDX]) / bin_wise_count
            downsampled_data[:, Cgbdisagg.INPUT_SPC_HUM_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_SPC_HUM_IDX]) / bin_wise_count
            downsampled_data[:, Cgbdisagg.INPUT_REL_HUM_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_REL_HUM_IDX]) / bin_wise_count
            downsampled_data[:, Cgbdisagg.INPUT_WET_BULB_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_WET_BULB_IDX]) / bin_wise_count
            downsampled_data[:, Cgbdisagg.INPUT_WIND_DIR_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_WIND_DIR_IDX]) / bin_wise_count

            downsampled_data[:, Cgbdisagg.INPUT_VISIBILITY_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_VISIBILITY_IDX]) / bin_wise_count
            downsampled_data[:, Cgbdisagg.INPUT_COOLING_POTENTIAL_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_COOLING_POTENTIAL_IDX]) / bin_wise_count
            downsampled_data[:, Cgbdisagg.INPUT_HEATING_POTENTIAL_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_HEATING_POTENTIAL_IDX]) / bin_wise_count
            downsampled_data[:, Cgbdisagg.INPUT_WH_POTENTIAL_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_WH_POTENTIAL_IDX]) / bin_wise_count
            downsampled_data[:, Cgbdisagg.INPUT_COLD_EVENT_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_COLD_EVENT_IDX]) / bin_wise_count
            downsampled_data[:, Cgbdisagg.INPUT_HOT_EVENT_IDX] = \
                np.bincount(bin_idx, weights=input_data[:, Cgbdisagg.INPUT_HOT_EVENT_IDX]) / bin_wise_count

        return downsampled_data

    else:
        return input_data
