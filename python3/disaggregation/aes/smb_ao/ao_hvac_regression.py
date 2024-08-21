"""
Author: Abhinav Srivastava
Date:   08-Feb-2020
Computing AO HVAC for SMB [Function separated and moved to this new file by Neelabh on 14 June 2023]
"""

# Import python packages

import numpy as np
import scipy
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Import functions from within the project

from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params


def get_valid_degree_days(degree_day, day_ao):
    """
    Function to get valid degree days by removing outliers, just like DS-450

    Parameters:

        degree_day (np.ndarray)         : Array of cdd or hdd
        day_ao (np.ndarray)             : Array of day level AO

    Returns:
        valid_degree_day (np.ndarray)   : Array of valid cdd or hdd points added
    """

    static_params = hvac_static_params()

    valid_degree_day = scipy.ones(np.size(day_ao), float)
    valid_degree_day[degree_day <= 0] = 0
    valid_degree_day[day_ao <= 0] = 0

    degree_day_max = np.mean(degree_day[degree_day > 0]) + 0.5 * np.std(degree_day[degree_day > 0])
    degree_day_min = np.mean(degree_day[degree_day > 0]) - 0.5 * np.std(degree_day[degree_day > 0])

    day_ao_max = np.mean(day_ao[day_ao > 0]) + 0.5 * np.std(day_ao[day_ao > 0])
    day_ao_min = np.mean(day_ao[day_ao > 0]) - 0.5 * np.std(day_ao[day_ao > 0])

    no_degree_day_1 = np.sum((day_ao < day_ao_min) & (degree_day > degree_day_max))
    no_degree_day_2 = np.sum((day_ao > day_ao_max) & (degree_day < degree_day_min))

    max_days_to_remove = static_params['ao']['max_days_to_remove']

    if no_degree_day_1 <= max_days_to_remove:
        valid_degree_day[(day_ao < day_ao_min) & (degree_day > degree_day_max)] = 0

    if no_degree_day_2 <= max_days_to_remove:
        valid_degree_day[(day_ao > day_ao_max) & (degree_day < degree_day_min)] = 0

    return valid_degree_day


def get_ao_cooling_regression(cdd, valid_cdd, day_ao_over_baseload, disagg_output_object, logger_ao):
    """
    Function to perform regression for cooling

    Parameters:

        cdd (np.ndarray)                    : Array of cdd with base of 65 F
        valid_cdd (np.ndarray)              : Array of valid cooling points
        day_ao_over_baseload (np.ndarray)   : Array of ao that is over baseload calculated earlier
        disagg_output_object (dict)         : Dictionary containing all the outputs
        logger_ao (logging object)          : Records progress of algo

    Returns:

        ao_cooling_regression (np.ndarray)  : Array of cooling extracted from regression
    """

    # noinspection PyBroadException
    try:

        logger_ao.info(' Attempting cooling regression. |')
        cooling_regression = LinearRegression().fit(cdd[valid_cdd == 1].reshape(-1, 1),
                                                    day_ao_over_baseload[valid_cdd == 1].reshape(-1, 1))
        cooling_coefficient = cooling_regression.coef_

        if cooling_coefficient[0][0] <= 0:
            cooling_coefficient = np.array([[0]])

        cool_r_square = pearsonr(cdd[valid_cdd == 1], day_ao_over_baseload[valid_cdd == 1])[0]
        ao_cooling_regression = list(cdd * cooling_coefficient[0])

        logger_ao.info(
            ' Regression part of AO Cooling. Coefficient : {}, R-sq : {} |'.format(cooling_coefficient, cool_r_square))
        disagg_output_object['created_hsm']['ao']['attributes']['ac_coefficient'] = cooling_coefficient[0]

    except Exception:

        logger_ao.info(' By-passing regression. Not enough valid points for ao regression |')
        ao_cooling_regression = list(np.repeat(0, len(cdd)))
        disagg_output_object['created_hsm']['ao']['attributes']['ac_coefficient'] = 0

    return ao_cooling_regression


def get_ao_heating_regression(hdd, valid_hdd, day_ao_over_baseload, disagg_output_object, logger_ao):
    """
    Function to perform regression for heating

    Parameters:

        hdd (np.ndarray)                    : Array of hdd with base of 65 F
        valid_hdd (np.ndarray)              : Array of valid heating points
        day_ao_over_baseload (np.ndarray)   : Array of ao that is over baseload calculated earlier
        disagg_output_object (dict)         : Dictionary containing all the outputs
        logger_ao (logging object)          : Records progress of algo

    Returns:

        ao_heating_regression (np.ndarray)  : Array of heating extracted from regression
    """

    # noinspection PyBroadException
    try:

        logger_ao.info(' Attempting heating regression. |')
        heating_regression = LinearRegression().fit(hdd[valid_hdd == 1].reshape(-1, 1),
                                                    day_ao_over_baseload[valid_hdd == 1].reshape(-1, 1))
        heating_coefficient = heating_regression.coef_

        if heating_coefficient[0][0] <= 0:
            heating_coefficient = np.array([[0]])

        heat_r_square = pearsonr(hdd[valid_hdd == 1], day_ao_over_baseload[valid_hdd == 1])[0]
        ao_heating_regression = list(hdd * heating_coefficient[0])

        logger_ao.info(
            ' Regression part of AO Heating. Coefficient : {}, R-sq : {} |'.format(heating_coefficient, heat_r_square))
        disagg_output_object['created_hsm']['ao']['attributes']['sh_coefficient'] = heating_coefficient[0]

    except Exception:

        logger_ao.info(' By-passing regression. Not enough valid points for ao regression |')
        ao_heating_regression = list(np.repeat(0, len(hdd)))
        disagg_output_object['created_hsm']['ao']['attributes']['sh_coefficient'] = 0

    return ao_heating_regression
