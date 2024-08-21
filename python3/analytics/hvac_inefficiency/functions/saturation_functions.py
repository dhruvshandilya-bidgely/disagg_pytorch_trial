"""
Author  -   Anand Kumar Singh
Date    -   22th Feb 2021
This file contains code for finding saturation temperature
"""

# Import python packages

import copy
import logging
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

# Import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff
from python3.analytics.hvac_inefficiency.utils.metrics import median_absolute_error
from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile as super_percentile


def get_pre_saturation(arr_x, arr_y, min_duty_cycle_limit, average_duty_cycle_limit, logger_pass, device='ac'):

    """
        Get pre saturation temperature based on temperature and duty cycle relationship

        Parameters:
            arr_x                           (numpy.ndarray)    : array of integral temperature
            arr_y                           (numpy.ndarray)    : array of integral duty cycle
            min_duty_cycle_limit            (float)            : limit on minimum duty cycle
            average_duty_cycle_limit        (float)            : limit on average duty cycle
            logger_pass                     (logger)           : logger object
            device                          (str)              : consumption device

        Returns:
            pre_saturation_temperature      (int/str)          : pre saturation temperature
    """

    # Taking new logger base for this module

    logger_local = logger_pass.get("logger").getChild("get_pre_saturation")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}

    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    get_pre_sat_time = datetime.datetime.now()

    # creating array to store relationship

    temp_dc_relation = np.c_[arr_x, arr_y]

    low_quantile_value = 0.20
    temperature_col = 0
    duty_cycle_col = 1
    step_size = 1

    # start and end temperature in the relations

    start_temperature = int(np.nanmin(temp_dc_relation[:, temperature_col]))
    end_temperature = int(np.nanmax(temp_dc_relation[:, temperature_col]))

    pre_saturation_condn_list = []

    logger.debug('Iterating over temperature values to find condition match |')

    for temperature_value in range(start_temperature, end_temperature, step_size):
        if device == 'ac':
            logger.debug('{} finding temperatures greater than | {}'.format(device, temperature_value))
            valid_idx = temp_dc_relation[:, temperature_col] >= temperature_value
        else:
            logger.debug('{} finding temperatures lesser than | {}'.format(device, temperature_value))
            valid_idx = temp_dc_relation[:, temperature_col] <= temperature_value

        average_duty_cycle = np.nanmedian(temp_dc_relation[valid_idx, duty_cycle_col])
        minimum_duty_cycle = super_percentile(temp_dc_relation[valid_idx, duty_cycle_col], low_quantile_value * 100)

        pre_saturation_condn_list.append([temperature_value, average_duty_cycle, minimum_duty_cycle])

    # Creating a numpy array
    pre_saturation_condn_list = np.array(pre_saturation_condn_list)

    min_duty_cycle_col = 2
    average_duty_cycle_col = 1

    pre_saturation_temperature = 'Unknown'

    if pre_saturation_condn_list.shape[0] == 0:
        logger.debug('{} Not enough temperature to compute |')
        pre_saturation_temperature = "Not enough temperature with {}".format(device)

    else:

        # Checking average and minimum condition indices

        minimum_condition_idx = pre_saturation_condn_list[:, min_duty_cycle_col] > min_duty_cycle_limit
        average_condition_idx = pre_saturation_condn_list[:, average_duty_cycle_col] > average_duty_cycle_limit
        final_condition_idx = average_condition_idx & minimum_condition_idx

        if pre_saturation_condn_list[final_condition_idx, temperature_col].shape[0] == 0:
            logger.debug('Final pre saturation condition does not hold for any temperature |')
            pre_saturation_temperature = "Duty Cycle Limits Failed"

        else:

            pre_saturation_offset = 0
            temperature_condn = False
            condition_indication = None
            valid_temperature_count = 0

            if device == 'ac':
                pre_saturation_temperature = (pre_saturation_condn_list[final_condition_idx, temperature_col]).min()
                pre_saturation_temperature = int(pre_saturation_temperature)
                condition_indication = 'max'
                temperature_condn = pre_saturation_temperature == pre_saturation_condn_list[:, temperature_col].max()
                valid_idx = pre_saturation_condn_list[:, temperature_col] > pre_saturation_temperature
                pre_saturation_offset = 3
                valid_temperature_count = valid_idx.sum()

            elif device == 'sh':
                pre_saturation_temperature = (pre_saturation_condn_list[final_condition_idx, temperature_col]).max()
                pre_saturation_temperature = int(pre_saturation_temperature)
                condition_indication = 'min'
                temperature_condn = pre_saturation_temperature == pre_saturation_condn_list[:, temperature_col].min()
                pre_saturation_offset = -3
                valid_idx = pre_saturation_condn_list[:, temperature_col] < pre_saturation_temperature
                valid_temperature_count = valid_idx.sum()

            if temperature_condn:

                logger.debug('Pre saturation temperature same as {} temperature|'.format(condition_indication))
                pre_saturation_temperature = "Pre Saturation at {} temperature".format(condition_indication)

            elif valid_temperature_count <= 3:
                logger.debug('Did not find enough temperature points beyond pre-saturation |')
                pre_saturation_temperature = 'Not enough temperature beyond pre-saturation'

            else:
                pre_saturation_temperature += pre_saturation_offset

    time_taken = get_time_diff(get_pre_sat_time, datetime.datetime.now())
    logger.debug('Time taken for getting pre saturation temperature | {} | {}'.format(device, time_taken))

    return pre_saturation_temperature


def get_saturation_temperature_flat(device, valid_saturated_temperature, slope_list, final_condition_idx):

    """
    Function to get saturation temperature

    Parameters:
        device                              (str)           : HVAC Device
        valid_saturated_temperature         (int)           : Failsafe saturation temperature
        slope_list                          (np.ndarray)    : list of slopes
        final_condition_idx                 (int)           : Final condition idx

    Returns:
        saturation_temperature              (int)   : Saturation temperature
    """

    saturation_temperature = -100

    if (device == 'ac') & (valid_saturated_temperature > 0):
        saturation_temperature = slope_list[final_condition_idx, 0].min()

    elif (device == 'sh') & (valid_saturated_temperature > 0):
        saturation_temperature = slope_list[final_condition_idx, 0].max()

    return saturation_temperature


def get_slope_list(slope, updated_test_list, idx, median_error, slope_list, temp_array):

    """
    Function to get slope array

    Parameters:
        slope               : Slope
        updated_test_list   : Updated test list
        idx                 : Index
        median_error        : Median error
        slope_list          : Slope list
        temp_array          : Temp array

    Returns:
        temp_array          : Temp array
        slope_list          : Slope list
    """

    if slope is not None:
        temp_array = np.array([[updated_test_list[idx, 0], slope, median_error]])
        slope_list = np.r_[slope_list, temp_array]

    return temp_array, slope_list


def get_flat_saturation(updated_test_list, device):

    """
        Get pre saturation flatness condition

        Parameters:
            updated_test_list           (numpy.ndarray)     limit on average duty cycle
            device                      (str)               consumption device

        Returns:
            valid_saturated_temperature (int)               number of temperature points with valid saturation
            saturation_temperature      (int)               saturation temperature
    """

    slope_list = np.empty((0, 3))
    slope = None
    median_error = None

    slope_threshold = 0.09
    median_error_threshold = 0.03

    range_temperature = updated_test_list[:, 0].max() - updated_test_list[:, 0].min()
    min_temperature = updated_test_list[:, 0].min()

    temp_array = np.array([])

    if device == 'ac':
        for idx in range(updated_test_list.shape[0]):
            regressor = LinearRegression()
            regression_array = copy.deepcopy(updated_test_list[idx:, :])
            regression_array[:, 0] = (regression_array[:, 0] - min_temperature) / range_temperature

            if regression_array.shape[0] <= 3:

                temp_array, slope_list = get_slope_list(slope, updated_test_list, idx, median_error, slope_list,
                                                        temp_array)

                continue

            regressor.fit(regression_array[:, 0].reshape(-1, 1), regression_array[:, 1].reshape(-1, 1))
            regressor_output = regressor.predict(regression_array[:, 0].reshape(-1, 1))
            median_error = median_absolute_error(regression_array[:, 1].reshape(-1, 1), regressor_output)
            slope = np.arctan(regressor.coef_[0])
            slope = slope[0]

            temp_array = np.array([[updated_test_list[idx, 0], slope, median_error]])
            slope_list = np.r_[slope_list, temp_array]

    elif device == 'sh':
        for idx in range((updated_test_list.shape[0] - 1), -1, -1):
            regressor = LinearRegression()
            regression_array = copy.deepcopy(updated_test_list[:idx, :])

            regression_array[:, 0] = (regression_array[:, 0] - min_temperature) / range_temperature

            if regression_array.shape[0] <= 3:

                temp_array, slope_list = get_slope_list(slope, updated_test_list, idx, median_error, slope_list,
                                                        temp_array)

                continue

            regressor.fit(regression_array[:, 0].reshape(-1, 1), regression_array[:, 1].reshape(-1, 1))
            regressor_output = regressor.predict(regression_array[:, 0].reshape(-1, 1))
            median_error = median_absolute_error(regression_array[:, 1].reshape(-1, 1), regressor_output)
            slope = np.arctan(regressor.coef_[0])
            slope = slope[0]

            temp_array = np.array([[updated_test_list[idx, 0], slope, median_error]])
            slope_list = np.r_[slope_list, temp_array]

    slope_col = 1
    median_error_col = 2

    final_condition_idx = (np.abs(slope_list[:, slope_col]) <= slope_threshold) & \
                          (np.abs(slope_list[:, median_error_col]) <= median_error_threshold)

    valid_saturated_temperature = final_condition_idx.sum()

    saturation_temperature = get_saturation_temperature_flat(device, valid_saturated_temperature, slope_list,
                                                             final_condition_idx)

    return valid_saturated_temperature, saturation_temperature


def get_ac_saturation_condition(temp_dc_relation, temperature_col, start_temperature, duty_cycle_col,
                                low_quantile_value, saturation_condn_list):
    """
    Function to get ac saturation condition

    Parameters:

        temp_dc_relation                : Temp dc relation
        temperature_col                 : Temperature col
        start_temperature               : Start temperature
        duty_cycle_col                  : Duty cycle col
        low_quantile_value              : Low quantile value
        saturation_condn_list           : Saturation condn list

    Returns:
        saturation_condn_list (list)    : Saturation condn list
    """

    step_size = 1
    end_temperature = int(np.nanmax(temp_dc_relation[:, temperature_col]))

    for temperature_value in range(start_temperature, end_temperature, step_size):
        valid_idx = temp_dc_relation[:, temperature_col] >= temperature_value
        average_duty_cycle = np.nanmedian(temp_dc_relation[valid_idx, duty_cycle_col])
        minimum_duty_cycle = super_percentile(temp_dc_relation[valid_idx, duty_cycle_col], low_quantile_value * 100)
        saturation_condn_list.append([temperature_value, average_duty_cycle, minimum_duty_cycle])

    return saturation_condn_list


def get_sh_saturation_condition(temp_dc_relation, temperature_col, start_temperature, duty_cycle_col,
                                low_quantile_value, saturation_condn_list):
    """
    Function to get ac saturation condition

    Parameters:

        temp_dc_relation                : Temp dc relation
        temperature_col                 : Temperature col
        start_temperature               : Start temperature
        duty_cycle_col                  : Duty cycle col
        low_quantile_value              : Low quantile value
        saturation_condn_list           : Saturation condn list

    Returns:
        saturation_condn_list (list)    : Saturation condn list
    """

    step_size = -1
    end_temperature = int(np.nanmin(temp_dc_relation[:, temperature_col]))

    for temperature_value in range(start_temperature, end_temperature, step_size):
        valid_idx = temp_dc_relation[:, temperature_col] <= temperature_value
        average_duty_cycle = np.nanmedian(temp_dc_relation[valid_idx, duty_cycle_col])
        minimum_duty_cycle = super_percentile(temp_dc_relation[valid_idx, duty_cycle_col], low_quantile_value * 100)
        saturation_condn_list.append([temperature_value, average_duty_cycle, minimum_duty_cycle])

    return saturation_condn_list


def get_sat_condn_list(temp_dc_relation, temperature_col, start_temperature, duty_cycle_col,
                       low_quantile_value, saturation_condn_list, device):
    """
    Function to get ac saturation condition

    Parameters:

        temp_dc_relation                : Temp dc relation
        temperature_col                 : Temperature col
        start_temperature               : Start temperature
        duty_cycle_col                  : Duty cycle col
        low_quantile_value              : Low quantile value
        saturation_condn_list           : Saturation condn list

    Returns:
        saturation_condn_list (list)    : Saturation condn list
    """

    if device == 'ac':

        saturation_condn_list = get_ac_saturation_condition(temp_dc_relation, temperature_col, start_temperature,
                                                            duty_cycle_col, low_quantile_value, saturation_condn_list)

    elif device == 'sh':

        saturation_condn_list = get_sh_saturation_condition(temp_dc_relation, temperature_col, start_temperature,
                                                            duty_cycle_col, low_quantile_value, saturation_condn_list)

    return saturation_condn_list


def get_saturation_temp_ac(saturation_condn_list, final_condition_idx):

    """
    Function to get ac saturation temperature

    Parameters:
        saturation_condn_list         : Saturation condn list
        final_condition_idx           : condition index
    Returns:
        saturation_temperature        : Saturation temperature
    """

    if saturation_condn_list[~final_condition_idx, 0].shape[0] == 0:
        saturation_temperature = saturation_condn_list[:, 0].min()
    else:
        saturation_temperature = (saturation_condn_list[~final_condition_idx, 0]).max()

    return saturation_temperature


def get_saturation_temp_sh(saturation_condn_list, final_condition_idx):

    """
    Function to get ac saturation temperature

    Parameters:
        saturation_condn_list         : Saturation condn list
        final_condition_idx           : condition index
    Returns:
        saturation_temperature        : Saturation temperature
    """

    if saturation_condn_list[~final_condition_idx, 0].shape[0] == 0:
        saturation_temperature = saturation_condn_list[:, 0].max()
    else:
        saturation_temperature = (saturation_condn_list[~final_condition_idx, 0]).min()

    return saturation_temperature


def get_saturation(x, y, min_duty_cycle_limit, average_duty_cycle_limit, pre_saturation_temp, logger_pass, device):

    """
        Get pre saturation temperature based on temperature and duty cycle relationship

        Parameters:
            x                           (numpy.ndarray)     array of integral temperature
            y                           (numpy.ndarray)     array of integral duty cycle
            min_duty_cycle_limit        (float)             limit on minimum duty cycle
            pre_saturation_temp         (int)               pre saturation temperature defined above
            average_duty_cycle_limit    (float)             limit on average duty cycle
            logger_pass                 (object)            logger object
            device                      (str)               consumption device

        Returns:
            saturation_temperature      (int/str)           pre saturation temperature
    """

    # Taking new logger base for this module

    logger_local = logger_pass.get("logger").getChild("get_saturation")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}

    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    get_sat_time = datetime.datetime.now()

    temp_dc_relation = np.c_[x, y]

    duty_cycle_col = 1
    temperature_col = 0
    low_quantile_value = 0.20

    # Initialise  starting and ending point

    start_temperature = pre_saturation_temp

    saturation_condn_list = []

    # Prepare list of min and average duty cycle
    logger.debug('Checking saturation condition for all the temperature values |')

    saturation_condn_list = get_sat_condn_list(temp_dc_relation, temperature_col, start_temperature, duty_cycle_col,
                                               low_quantile_value, saturation_condn_list, device)

    saturation_condn_list = np.array(saturation_condn_list)

    min_duty_cycle_col = 2
    average_duty_cycle_col = 1

    average_condition_idx = saturation_condn_list[:, average_duty_cycle_col] > average_duty_cycle_limit
    minimum_condition_idx = saturation_condn_list[:, min_duty_cycle_col] > min_duty_cycle_limit
    final_condition_idx = average_condition_idx & minimum_condition_idx

    offset = 0
    temperature_condn = False
    condition_indication = None

    if saturation_condn_list[final_condition_idx, temperature_col].shape[0] == 0:
        logger.debug('Duty cycle limits failed for all temperature points |')
        saturation_temperature = "Duty Cycle Limits Failed"

    else:
        if device == 'ac':

            saturation_temperature = get_saturation_temp_ac(saturation_condn_list, final_condition_idx)

            offset = 2
            condition_indication = 'max'
            final_condition_idx = temp_dc_relation[:, 0] >= saturation_temperature
            temperature_condn = (saturation_temperature == saturation_condn_list[:, 0].max())

        elif device == 'sh':

            saturation_temperature = get_saturation_temp_sh(saturation_condn_list, final_condition_idx)

            offset = (-2)
            condition_indication = 'min'
            final_condition_idx = temp_dc_relation[:, 0] <= saturation_temperature
            temperature_condn = saturation_temperature == saturation_condn_list[:, 0].min()

        updated_test_list = temp_dc_relation[final_condition_idx, :]

        if temperature_condn:

            logger.debug(' Saturation at {} temperature |'.format(condition_indication))

            saturation_temperature = "Saturation at {} temperature".format(condition_indication)

        else:

            valid_saturated_temperature, saturation_temperature = get_flat_saturation(updated_test_list, device)

            saturation_temperature += offset
            saturation_temperature = int(saturation_temperature)

            if valid_saturated_temperature < 3:
                saturation_temperature = "Not enough flatness"

    time_taken = get_time_diff(get_sat_time, datetime.datetime.now())
    logger.debug('Time taken for getting saturation temperature | {} | {}'.format(device, time_taken))

    return saturation_temperature
