"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module to categorize data points into different season
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.ev.functions.ev_utils import bill_cycle_to_month
from python3.disaggregation.aer.ev.functions.ev_utils import find_missing_bill_cycle_season

from python3.disaggregation.aer.ev.functions.get_seasons_cutoff import get_seasons_cutoff

# List of columns and their corresponding indices used in seasonal info

season_columns = {
    'bill_cycle_ts': 0,
    'average_temp': 1,
    'season_id': 2,
}


def get_season_categorisation(input_data, setpoint=65):
    """
    This functions divide data into summer, winter, and transition based on temperature

    Parameters:
        input_data      (np.ndarray)    : Input 21-column matrix
        setpoint        (int)           : Default setpoint to find seasons

    Returns:
        wtr_tuple       (tuple)         : Winter data and indices
        itr_tuple       (tuple)         : Transition data and indices
        smr_tuple       (tuple)         : Summer data and indices
        all_seasons     (np.ndarray)    : Seasonal information
    """
    # Consider only non-NAN temperatures

    temp_data = deepcopy(input_data)
    temp_data = temp_data[~np.isnan(temp_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX])]

    # Get unique bill cycle timestamps and corresponding indices

    unique_months, months_idx, months_count = np.unique(temp_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                                                        return_counts=True,
                                                        return_inverse=True)

    # Calculate monthly average temperature

    avg_temp = np.bincount(months_idx, temp_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]) / months_count
    avg_temp = np.round(avg_temp, 3)

    # Stack bill cycle timestamps and temperature together

    seasons_info = np.hstack((unique_months.reshape(-1, 1), np.round(avg_temp.reshape(-1, 1), 3),
                              np.empty((len(avg_temp), 1))))

    # Find season for bill cycle with no valid temperature value

    seasons_info = find_missing_bill_cycle_season(input_data, seasons_info)

    # Sort the season data based on average monthly temperature

    seasons_info = seasons_info[np.lexsort((seasons_info[:, season_columns.get('bill_cycle_ts')],
                                            seasons_info[:, season_columns.get('average_temp')]))]

    # Check difference of average temperature with respect to set point

    temp_diff = np.abs(seasons_info[:, season_columns['average_temp']] - setpoint)

    # Get the allowed number of transition months from config

    n_transitions = np.int(np.ceil(len(temp_diff) / 4))

    # Find the transition month index

    if len(temp_diff) > n_transitions:
        # If number of months more than required number of transition months

        transition_limit = sorted(temp_diff)[n_transitions]
    else:
        # If number of months less than required number of transition months

        transition_limit = setpoint

    # Retrieve the cutoff value's index

    transition_idx = np.where(temp_diff <= transition_limit)[0]

    # Get the temperature bounds for winter and summer

    wtr_cutoff, smr_cutoff = get_seasons_cutoff(seasons_info, season_columns, transition_idx, unique_months,
                                                one_season=False, width=6)

    # Separating winter season segment

    wtr_temp_seg = seasons_info[seasons_info[:, season_columns['average_temp']] < wtr_cutoff, :]
    wtr_temp_seg[:, season_columns['season_id']] = 0

    # Separating intermediate season segment

    itr_temp_seg = seasons_info[(seasons_info[:, season_columns['average_temp']] >= wtr_cutoff) &
                                (seasons_info[:, season_columns['average_temp']] <= smr_cutoff), :]
    itr_temp_seg[:, season_columns['season_id']] = 1

    # Separating summer season segment

    smr_temp_seg = seasons_info[seasons_info[:, season_columns['average_temp']] > smr_cutoff, :]
    smr_temp_seg[:, season_columns['season_id']] = 2

    # Separating indices for each season in the input data

    wtr_idx = np.in1d(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], wtr_temp_seg[:, season_columns['bill_cycle_ts']])
    itr_idx = np.in1d(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], itr_temp_seg[:, season_columns['bill_cycle_ts']])
    smr_idx = np.in1d(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], smr_temp_seg[:, season_columns['bill_cycle_ts']])

    return wtr_idx, itr_idx, smr_idx


def add_season_month_tag(input_data):
    """
    This functions adds calendar month number and season to every epoch in 21-column matrix

    Parameters:
        input_data      (np.ndarray)    : Input 21-column matrix

    Returns:
        input_data      (np.ndarray)    : Modified 21-column matrix

    """
    # getting month number and season epoch level

    input_data_copy = bill_cycle_to_month(input_data)

    uniq_months, uniq_month_indices = np.unique(input_data_copy[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_inverse=True)
    input_data = np.c_[input_data, uniq_month_indices]

    wtr_idx, itr_idx, smr_idx = get_season_categorisation(input_data, setpoint=65)

    seasons_tagged = np.zeros((input_data.shape[0],))
    seasons_tagged[wtr_idx] = 0
    seasons_tagged[itr_idx] = 1
    seasons_tagged[smr_idx] = 2

    input_data = np.c_[input_data, seasons_tagged]

    return input_data
