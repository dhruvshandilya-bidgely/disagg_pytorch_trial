"""
Author - Anand Kumar Singh
Date - 26th June 2021
HVAC inefficiency HSM related operations
"""

# Import python packages

import numpy as np


def get_temperature_dc_relationship(temperature, duty_cycle):

    """
        Get temperature vs duty cycle relationship

        Parameters:
            temperature             (numpy.ndarray)          array containing temperature values
            duty_cycle              (numpy.ndarray)          array containing duty cycles
        Returns:
            duty_cycle_relation     (numpy.ndarray)          array containing temperature and corresponding median dc
    """

    temperature_column = 0
    val_idx = (~np.isnan(duty_cycle)) & (~np.isnan(temperature))

    val_temp = temperature[val_idx].astype(int)
    val_dcma = duty_cycle[val_idx].astype(float)

    un_temp, un_temp_idx = np.unique(val_temp, return_inverse=True)

    med_arr = []
    len_arr = []

    for idx in range(len(un_temp)):
        val_data = val_dcma[un_temp_idx == idx]
        med_arr.append(np.median(val_data))
        len_arr.append(len(val_data))

    # Initialising column indices for new numpy array

    duty_cycle_relation = np.c_[un_temp, np.array(med_arr), np.array(len_arr)]
    duty_cycle_relation = duty_cycle_relation[~(duty_cycle_relation[:, temperature_column] == 0)]

    return duty_cycle_relation
