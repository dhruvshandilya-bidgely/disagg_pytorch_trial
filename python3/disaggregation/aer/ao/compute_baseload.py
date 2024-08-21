"""
Author - Abhinav Srivastava
Date - 10th Oct 2018
Computing Baseload
"""

# Import python packages
import copy
import scipy
import logging
import numpy as np
from scipy.stats.mstats import mquantiles

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params


def compute_baseload(input_data, sampling_rate, logger_base):

    """
    Function to compute ao at epoch level

    Parameters:

        input_data      (np.ndarray)        : input data matrix with 21 columns
        valid           (np.ndarray)        : logical array of valid epochs based on consumption and temperature
        sampling_rate   (int)               : The sampling rate of the data
        logger_base     (logger object)     : object to log important steps of always on code flow

    Returns:
        month_algo_baseload      (np.ndarray)    : always on estimate at month level
        epoch_algo_baseload      (np.ndarray)    : always on estimate at epoch level
        epoch_raw_minus_baseload (np.ndarray)    : consumption left at epoch level after ao is removed
        exit_status              (dict)          : dictionary containing exit status and error codes for ao module
    """

    logger_local = logger_base.get("logger").getChild("compute_baseload")
    logger_ao_bl = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    input_data_copy = copy.deepcopy(input_data)

    # if input data is empty, return fail-safe estimate of ao and wholehouse
    if input_data_copy.size == 0:
        baseload = np.array([])
        wholehouse = np.array([])

    # getting epoch timestamps to paste with final result
    epoch_data = input_data_copy[:, Cgbdisagg.INPUT_EPOCH_IDX]

    # total number of hours in input
    number_of_hours = input_data_copy.shape[0]

    # initialize epoch level baseload vector
    baseload = np.nan * scipy.ones(number_of_hours, float)

    # getting unique month epoch stamps, to be used as specific month identifier
    unique_months = scipy.unique(input_data_copy[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX])

    # epoch level net consumption in input data array incoming to ao module
    wholehouse = input_data_copy[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    logger_ao_bl.info('Getting one baseload value for total unique months | {}'.format(len(unique_months)))

    # finding ao for each month, one month at a time
    for unique_idx in range(len(unique_months)):

        logger_ao_bl.info('Attempting baseload value for month | {}'.format(unique_months[unique_idx]))

        # identifying epochs corresponding to concerned month and corresponding valid epochs
        month_idx = input_data_copy[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == unique_months[unique_idx]
        month_idx_and_valid = np.logical_and(month_idx, np.logical_not(np.isnan(wholehouse)))

        if any(month_idx_and_valid):

            # determine ao level, returns an epoch level vector
            baseload[month_idx_and_valid] = baseload_for_month(wholehouse[month_idx_and_valid], sampling_rate, unique_months[unique_idx], logger_ao_bl)
            # remove ao from input consumption, the remaining epoch consumption will flow in hvac module
            wholehouse[month_idx_and_valid] = wholehouse[month_idx_and_valid] - baseload[month_idx_and_valid]

    wholehouse[wholehouse < 0] = 0
    epoch_raw_minus_baseload = wholehouse

    month_epoch, idx_2, month_idx = scipy.unique(input_data_copy[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_index=True, return_inverse=True)

    # aggregating always on estimates at month level for monthly estimates
    baseload_cp = baseload.copy()
    baseload[np.isnan(baseload)] = 0
    baseload_cp[np.isnan(baseload_cp)] = 0

    month_baseload = np.bincount(month_idx, baseload_cp)
    logger_ao_bl.info('aggregated baseload estimates at month level |')

    # concatenating month level appliance estimates, to form consumption array
    month_algo_baseload = np.c_[month_epoch, month_baseload]
    logger_ao_bl.info('concatenated month level always on estimates to month index |')

    # concatenating the epoch timestamps adjacent to epoch level always on estimates, for output
    epoch_algo_baseload = np.c_[epoch_data, baseload]
    logger_ao_bl.info('concatenated epoch level always on estimates to epoch index |')

    # dictionary to store always on algo related error codes and exit status
    exit_status = {
        'exit_code': 1,
        'error_list': [],
    }

    return month_algo_baseload, epoch_algo_baseload, epoch_raw_minus_baseload, exit_status


def baseload_for_month(wholehouse, sampling_rate, bill_cycle, logger_ao):

    """
    Function to find the epoch level ao for a particular month

    Parameters:
        wholehouse          (np.ndarray)    : Array containing the total consumption numbers
        sampling_rate       (int)           : The sampling rate of the data
        bill_cycle          (int)           : The start timestamp of the bill cycle
        logger_ao           (logger)        : Logger object to log statements

    Returns:
        baseload            (np.ndarray)    : Baseload values containing array
    """

    static_params = hvac_static_params()
    ao_params = static_params.get('ao')

    minimum_baseload = ao_params['min_hour_baseload'] * sampling_rate / Cgbdisagg.SEC_IN_HOUR

    valid_idx = [wholehouse > minimum_baseload]
    valid_wholehouse = wholehouse[valid_idx]

    if len(valid_wholehouse) > ao_params.get('min_len_valid_points'):
        ao_value = mquantiles(valid_wholehouse, 0.01, alphap=0.5, betap=0.5)
    elif len(valid_wholehouse) > 0:
        ao_value = min(valid_wholehouse)
    else:
        ao_value = 0

    baseload = (wholehouse >= ao_value) * ao_value

    # Handling AO at valid indexes as per minimum baseload, but wholehouse value less than ao value
    indexes = ((wholehouse < ao_value) & (valid_idx))[0]
    baseload[indexes] = wholehouse[indexes]

    logger_ao.info('AO information | Bill cycle : %d, AO_value : %.2f, Num points : %d, Num valid points : %d, '
                   'Num points above ao_value: %d', int(bill_cycle), ao_value, wholehouse.shape[0],
                   valid_wholehouse.shape[0], np.sum(wholehouse >= ao_value))

    return baseload
