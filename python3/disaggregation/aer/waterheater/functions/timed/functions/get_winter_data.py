"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to retrieve the data of winter season
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_winter_data(in_data, in_box_data, logger, setpoint=65, width=9, transition=12):
    """
    Find the winter bill cycles and return the corresponding data

    Parameters:
        in_data         (np.ndarray)        : Input 21-column raw data
        in_box_data     (np.ndarray)        : Input 21-column box data
        logger          (logger)            : Logger object
        setpoint        (int)               : Default temperature setpoint
        width           (int)               : Default temperature deviation
        transition      (int)               : Allowed max deviation

    Returns:
        wtr_idx         (np.ndarray)        : Winter indices
        wtr_data        (np.ndarray)        : Winter raw data
        wtr_box_data    (np.ndarray)        : Winter box data
    """

    # Taking deepcopy of input data to keep local instances

    raw_data = deepcopy(in_data)
    box_data = deepcopy(in_box_data)

    # Consider only non-NaN temperatures

    temp_data = deepcopy(in_data)
    temp_data = temp_data[~np.isnan(temp_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX])]

    # Calculating the unique bill cycle timestamps and indices

    unq_months, months_idx, months_count = np.unique(temp_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                                                     return_counts=True,
                                                     return_inverse=True)

    # Calculate monthly average temperature

    avg_temp = np.bincount(months_idx, temp_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]) / months_count

    # Stack bill cycle timestamps and temperature together

    monthly_avg_temp = np.hstack((unq_months.reshape(-1, 1),
                                  np.round(avg_temp.reshape(-1, 1), 3)))

    # Sort the season data based on average monthly temperature

    monthly_avg_temp = monthly_avg_temp[np.lexsort((monthly_avg_temp[:, 0], monthly_avg_temp[:, 1]))]

    # Define the allowed temperature deviation

    wtr_maximum = setpoint - width

    # Check difference of average temperature with respect to set point

    temp_diff = np.abs(monthly_avg_temp[:, 1] - setpoint)

    # Find the transition month index

    if len(temp_diff) > 3:
        transition_limit = sorted(temp_diff)[3]
    else:
        transition_limit = transition

        logger.info('Using default transition limit | ')

    transition_idx = np.where(temp_diff <= transition_limit)[0]

    # Calculate the maximum temperature allowed for winter

    if len(transition_idx) > 0:
        wtr_cutoff = np.fmin(monthly_avg_temp[transition_idx[0], 1], wtr_maximum)
    else:
        wtr_cutoff = wtr_maximum

    logger.info('Winter temperature cutoff: | {}'.format(wtr_cutoff))

    # Separating winter season segment

    wtr_temp_seg = monthly_avg_temp[monthly_avg_temp[:, 1] <= wtr_cutoff, :]

    # Separating the data for winter season

    wtr_idx = np.in1d(raw_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], wtr_temp_seg[:, 0])

    # Return raw and box data along with indices for winter

    wtr_data = raw_data[wtr_idx, :]
    wtr_box_data = box_data[wtr_idx, :]

    return wtr_idx, wtr_data, wtr_box_data
