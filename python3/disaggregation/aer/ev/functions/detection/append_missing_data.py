"""
Author - Nikhil Singh Chauhan
Date - 15-May-2020
Module to append removed missing data
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def append_missing_data(debug):
    """
    Function to append missing data

        Parameters:
            debug           (dict)              : Dict containing hsm_in, bypass_hsm, make_hsm

        Returns:
            debug           (object)            : Object containing all important data/values as well as HSM
    """
    # Get the missing data

    missing_data = debug['input_data_missing']

    missing_data_shape = missing_data.shape

    if missing_data.shape[0] > 0:
        # Add missing data to baseload data

        baseload_data = debug.get('baseload_data')

        # Adding missing data to baseload

        baseload_data = np.vstack((baseload_data, missing_data))

        baseload_data = baseload_data[baseload_data[:, Cgbdisagg.INPUT_EPOCH_IDX].argsort()]

        debug['baseload_data'] = baseload_data

        # Adding missing data to input after baseload data

        input_after_baseload = debug.get('input_after_baseload')

        input_after_baseload = np.vstack((input_after_baseload, missing_data))

        input_after_baseload = input_after_baseload[input_after_baseload[:, Cgbdisagg.INPUT_EPOCH_IDX].argsort()]

        debug['input_after_baseload'] = input_after_baseload

        # Updating box data for all keys
        box_data_keys = [key for key in debug.keys() if 'box_data' in key]

        for data_key in box_data_keys:
            temp_box_data = debug.get(data_key)

            n_columns = temp_box_data.shape[1]

            # Add additional to box data if the number of columns in box data is different from that of missing dat
            if n_columns > missing_data_shape[1]:
                additional_columns = n_columns - missing_data_shape[1]

                temp_missing_data = np.hstack((missing_data,
                                               np.zeros(shape=(missing_data_shape[0], additional_columns))))
            else:
                temp_missing_data = missing_data

            temp_box_data = np.vstack((temp_box_data, temp_missing_data))

            temp_box_data = temp_box_data[temp_box_data[:, Cgbdisagg.INPUT_EPOCH_IDX].argsort()]

            # Adding updated box data to debug dictionary
            debug[data_key] = temp_box_data

    return debug
