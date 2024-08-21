"""
Author - Paras Tehria
Date - 17-Nov-2020
This module is used to run the get consumption features for solar propensity computation
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.maths_utils import forward_fill
from python3.utils.maths_utils.maths_utils import create_pivot_table


def get_consum_feat(in_data, solar_propensity_config):

    """
    This function computes consumption features for solar propensity

    Parameters:
        in_data              (np.ndarray)             : Input 21-columns data
        solar_propensity_config      (dict)            : solar propensity module parameters

    Return:
        daily_suntime_sum    (np.array)               : Array containing daily savings potential

    """

    input_data = deepcopy(in_data)

    # array contains one hot encoded array containing presence of sunlight

    sun_array = np.ones((len(input_data), 1), dtype=input_data.dtype)

    # Sunlight present when time is between sunrise and sunset times
    sun_presence = np.logical_and(
        input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] >= input_data[:, Cgbdisagg.INPUT_SUNRISE_IDX],
        input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] <= input_data[:, Cgbdisagg.INPUT_SUNSET_IDX])

    # Adding a one sun_array to act as new column

    sun_array[:, 0] = np.where(sun_presence, sun_array[:, 0], 0)

    nan_sunrise_sunset = np.logical_or(np.isnan(input_data[:, Cgbdisagg.INPUT_SUNRISE_IDX]),
                                       np.isnan(input_data[:, Cgbdisagg.INPUT_SUNSET_IDX]))
    sun_array[:, 0] = np.where(~nan_sunrise_sunset, sun_array[:, 0], np.nan)

    input_data = np.hstack((input_data, sun_array))

    y_signal_raw, _, _ = create_pivot_table(data=input_data, index=Cgbdisagg.INPUT_DAY_IDX,
                                            columns=Cgbdisagg.INPUT_HOD_IDX, values=Cgbdisagg.INPUT_CONSUMPTION_IDX)

    # Replacing na values by neighborhood days (ffill)
    y_signal_raw = forward_fill(y_signal_raw)

    # bfill
    y_signal_raw = np.flipud(forward_fill(np.flipud(y_signal_raw)))

    # Capping negative values to zero helps in capturing solar signals on normalised in_data

    y_signal_raw[y_signal_raw < 0] = 0

    # Generating solar presence pivot table
    sun_presence_col_idx = solar_propensity_config.get('sun_presence_col_idx')
    y_signal_sun, _, _ =\
        create_pivot_table(data=input_data, index=Cgbdisagg.INPUT_DAY_IDX,
                           columns=Cgbdisagg.INPUT_HOD_IDX, values=sun_presence_col_idx)

    # Replacing na values by neighborhood days (ffill)

    y_signal_sun = forward_fill(y_signal_sun)

    # bfill

    y_signal_sun = np.flipud(forward_fill(np.flipud(y_signal_sun)))

    # marking non-sunlight time consumption as 0
    cons_matrix = deepcopy(y_signal_raw)
    cons_matrix[~y_signal_sun.astype(bool)] = 0

    # rowwise sum of sunlight-time consumption
    daily_suntime_sum = np.nansum(cons_matrix, axis=1)

    return daily_suntime_sum
