"""
Author - Abhinav Srivastava
Date - 08-Feb-2020
Computing Baseload at Daily level for SMB
"""

# Import python packages

import copy
import scipy
import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile as super_percentile
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params


def get_hist_centers(data_per_hour, day_ao_min_limit, day_ao_max_limit):

    """
    Function returns the histogram centers for given range of values

    Parameters:
        data_per_hour       (int)           : Data points per hour
        day_ao_min_limit    (float)         : Minimum limit of day ao
        day_ao_max_limit    (float)         : Maximum limit of day ao

    Return:
        hist_centers        (np.ndarray)    : Histogram centers
    """

    static_params = hvac_static_params()
    ao_params = static_params['ao']

    if data_per_hour == ao_params.get('data_per_hour_30'):

        # histogram centers for 30 mins sampling
        hist_centers = np.arange(day_ao_min_limit, day_ao_max_limit, ao_params['hour_bin_size'] / ao_params['data_per_hour_30'])
        hist_centers = np.append(hist_centers, hist_centers[-1] + ao_params['hour_bin_size'] / ao_params['data_per_hour_30'])

    elif data_per_hour == static_params.get('ao').get('data_per_hour_15'):

        # histogram centers for 15 mins sampling
        hist_centers = np.arange(day_ao_min_limit, day_ao_max_limit, ao_params['hour_bin_size'] / ao_params['data_per_hour_15'])
        hist_centers = np.append(hist_centers, hist_centers[-1] + ao_params['hour_bin_size'] / ao_params['data_per_hour_15'])

    else:

        # histogram centers for general sampling
        hist_centers = np.arange(day_ao_min_limit, day_ao_max_limit, ao_params['hour_bin_size'])
        hist_centers = np.append(hist_centers, hist_centers[-1] + ao_params['hour_bin_size'])

    return hist_centers


def get_30_min_sampling_ao(epoch_count_so_far, consumption_so_far, bin_counts, index, day_ao, break_bool):
    """
    Get day ao and decision to break the loop for 30 min sampling

    Parameters:
        epoch_count_so_far      (int)       : Epoch counts till current loop
        consumption_so_far      (float)     : Consumption till current loop
        bin_counts              (np.ndarray): Histogram bin counts
        index                   (int)       : Current index
        day_ao                  (float)     : Day level ao
        break_bool              (bool)      : Decision to break the loop

    Return:
        day_ao                  (float)     : Day level ao
        break_bool              (bool)      : Decision to break the loop
    """

    static_params = hvac_static_params()
    ao_params = static_params.get('ao')

    if epoch_count_so_far >= ao_params.get('data_per_hour_30')+ 1:

        day_ao = np.sum(consumption_so_far) / np.sum(bin_counts[:index])
        break_bool = True

    return day_ao, break_bool


def get_15_min_sampling_ao(epoch_count_so_far, consumption_so_far, bin_counts, index, day_ao, break_bool):

    """
    Get day ao and decision to break the loop for 60 min sampling

    Parameters:
        epoch_count_so_far      (int)       : Epoch counts till current loop
        consumption_so_far      (float)     : Consumption till current loop
        bin_counts              (np.ndarray): Histogram bin counts
        index                   (int)       : Current index
        day_ao                  (float)     : Day level ao
        break_bool              (bool)      : Decision to break the loop

    Return:
        day_ao                  (float)     : Day level ao
        break_bool              (bool)      : Decision to break the loop
    """

    static_params = hvac_static_params()
    ao_params = static_params.get('ao')

    if epoch_count_so_far >= ao_params.get('data_per_hour_15') + 2:

        day_ao = np.sum(consumption_so_far) / np.sum(bin_counts[:index])
        break_bool = True

    return day_ao, break_bool


def get_general_sampling_ao(epoch_count_so_far, consumption_so_far, bin_counts, index, day_ao, break_bool):

    """
        Get day ao and decision to break the loop for general sampling

        Parameters:
            epoch_count_so_far      (int)       : Epoch counts till current loop
            consumption_so_far      (float)     : Consumption till current loop
            bin_counts              (np.ndarray): Histogram bin counts
            index                   (int)       : Current index
            day_ao                  (float)     : Day level ao
            break_bool              (bool)      : Decision to break the loop

        Return:
            day_ao                  (float)     : Day level ao
            break_bool              (bool)      : Decision to break the loop
        """

    if epoch_count_so_far > 1:

        day_ao = np.sum(consumption_so_far) / np.sum(bin_counts[:index])
        break_bool = True

    return day_ao, break_bool


def get_ao_for_day(bin_counts, hist_centers, data_per_hour):

    """
        Function to compute ao at day level

        Parameters:

            bin_counts (np.ndarray)         : input data matrix with 21 columns
            hist_centers (np.ndarray)       : object to log important steps of always on code flow
            data_per_hour (np.ndarray)      : dictionary containing user's config related parameters

        Returns:

            day_ao (np.ndarray)             : always on estimate at epoch level
        """

    static_params = hvac_static_params()
    ao_params = static_params.get('ao')

    # consumption initialization
    consumption_so_far = []
    epoch_count_so_far = 0

    # identifying ao value for current day
    for index in range(len(bin_counts)):

        # epoch and consumption count in loop so far
        epoch_count_so_far = epoch_count_so_far + bin_counts[index]
        consumption_so_far.append(bin_counts[index] * hist_centers[index])

        # initializing day ao as 0 and continuing loop
        day_ao = 0
        break_bool = False

        # getting histogram bin value according to data per hour
        if bin_counts[index] >= data_per_hour:

            day_ao = hist_centers[index]
            break_bool = True

        elif bin_counts[index] < data_per_hour:

            if data_per_hour == ao_params.get('data_per_hour_30'):

                # getting day ao at 30 min sampling
                day_ao, break_bool = get_30_min_sampling_ao(epoch_count_so_far, consumption_so_far, bin_counts, index, day_ao, break_bool)

            elif data_per_hour == ao_params.get('data_per_hour_15'):

                # getting day ao at 15 min sampling
                day_ao, break_bool = get_15_min_sampling_ao(epoch_count_so_far, consumption_so_far, bin_counts, index, day_ao, break_bool)

            else:

                # getting day ao at general sampling
                day_ao, break_bool = get_general_sampling_ao(epoch_count_so_far, consumption_so_far, bin_counts, index, day_ao, break_bool)

        # breaking search for day ao, if breaking condition is achieved
        if break_bool == True:
            break

    return day_ao


def compute_day_level_ao_smb(input_data, logger_base, global_config, disagg_input_object):

    """
    Function to compute ao at epoch level

    Parameters:

        input_data (np.ndarray)         : input data matrix with 21 columns
        logger_base (logger object)       : object to log important steps of always on code flow
        global_config (dict)            : dictionary containing user's config related parameters
        disagg_input_object (dict)      : dictionary containing all input related key information

    Returns:

        month_algo_ao (np.ndarray)      : contains month level ao values
        epoch_algo_ao (np.ndarray)      : always on estimate at epoch level
        epoch_raw_minus_ao (np.ndarray) : epoch level ao value removed from raw consumption
        exit_status (dict)              : dictionary containing exit status and error codes for ao module
    """

    logger_local = logger_base.get("logger").getChild("compute_day_level_ao_smb")
    logger_day_ao_smb = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    static_params = hvac_static_params()
    ao_params = static_params.get('ao')

    config = disagg_input_object.get('config')

    # getiing epoch timestamps and raw consumptions
    epoch_data = copy.deepcopy(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX])
    epoch_raw_data = copy.deepcopy(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # ensuring no nan value flow in baseload algo
    epoch_raw_data[np.isnan(epoch_raw_data)] = 0

    sampling_rate = global_config.get('sampling_rate')
    data_per_hour = Cgbdisagg.SEC_IN_HOUR / sampling_rate

    # initialize epoch level ao vector
    number_of_epochs = input_data.shape[0]
    epoch_ao = np.nan * scipy.ones(number_of_epochs, float)

    # getting unique month epoch stamps, to be used as specific month identifier
    unique_days = scipy.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX])
    logger_day_ao_smb.info(' Getting AO values for {} unique days |'.format(len(unique_days)))

    # finding ao value for each day. Average consumption that constitutes lowest 1 hour of consumption on a day.
    for day in unique_days:

        # identifying epochs corresponding to concerned day and getting only valid epochs from the concerned day
        day_idx = input_data[:, Cgbdisagg.INPUT_DAY_IDX] == day
        day_idx_and_valid = np.logical_and(day_idx, epoch_raw_data > 0)
        day_data = input_data[day_idx_and_valid, :]

        if len(day_data) >= 0.5 * (data_per_hour * Cgbdisagg.HRS_IN_DAY):

            day_ao_min_limit = float(np.min(day_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))
            day_ao_max_limit = float(np.around(super_percentile(day_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], 100), 5))

            # noinspection PyBroadException
            try:

                hist_centers = get_hist_centers(data_per_hour, day_ao_min_limit, day_ao_max_limit)
                bin_counts, _ = np.histogram(day_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], bins=hist_centers)
                day_ao = get_ao_for_day(bin_counts, hist_centers, data_per_hour)

            except (ValueError, IndexError, KeyError):

                day_ao = (day_ao_min_limit + day_ao_max_limit) / 2
                epoch_ao[day_idx] = day_ao

        else:

            day_ao = 0

        # Getting rid of Day AO where more than 20% data points are zeros
        day_non_zeros = len(day_data)
        day_size = np.sum(day_idx)
        cut_off_threshold = ao_params.get('day_zero_threshold')

        if (config.get('user_type').lower() == 'smb') & ((day_non_zeros / day_size) < cut_off_threshold):
            day_ao = 0

        epoch_ao[day_idx] = day_ao

    logger_day_ao_smb.info(' Describe day AO. sum : {}, min : {}, max : {} |'.format(np.nansum(epoch_ao), np.nanmin(epoch_ao), np.nanmax(epoch_ao)))

    logger_day_ao_smb.info(' Epochs where (AO > Net) : {} |'.format(np.nansum(epoch_ao > epoch_raw_data)))
    epoch_ao[epoch_ao > epoch_raw_data] = epoch_raw_data[epoch_ao > epoch_raw_data]
    logger_day_ao_smb.info(' Describe AO after handling (AO > net). sum : {}, min : {}, max : {} |'.format(np.nansum(epoch_ao), np.nanmin(epoch_ao), np.nanmax(epoch_ao)))

    epoch_raw_minus_ao = epoch_raw_data - epoch_ao

    exit_status = {
        'exit_code': 0,
        'error_list': [],
    }

    month_epoch, idx_2, month_idx = scipy.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_index=True, return_inverse=True)

    # aggregating always on estimates at month level for monthly estimates
    month_ao = np.bincount(month_idx, epoch_ao)
    logger_day_ao_smb.info(' Aggregated always on estimates at month level for monthly estimates |')

    # concatenating month level appliance estimates, to form consumption array
    month_algo_ao = np.c_[month_epoch, month_ao]
    logger_day_ao_smb.info(' Concatenated month level always on estimates, to form consumption array |')

    # concatenating the epoch timestamps adjacent to epoch level always on estimates, for output
    epoch_algo_ao = np.c_[epoch_data, epoch_ao]

    return month_algo_ao, epoch_algo_ao, epoch_raw_minus_ao, exit_status
