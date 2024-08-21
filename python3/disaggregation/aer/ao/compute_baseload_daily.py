"""
Author - Abhinav Srivastava
Date - 8th Feb 2019
Computing Baseload at Daily level for HVAC 3
"""

# Import python packages

import scipy
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile as super_percentile
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params


def get_hist_centers(data_per_hour, day_ao_min_limit, day_ao_max_limit):

    """
    Function to get histogram centers

    Parameters:
        data_per_hour       (int)           : Data sampling per hour
        day_ao_min_limit    (float)         : Minimum limit of day's ao
        day_ao_max_limit    (float)         : Maximum limit of day's ao:

    Returns:
        hist_centers        (np.ndarray)    : Histogram centers

    """

    static_params = hvac_static_params()
    ao_params = static_params['ao']

    if data_per_hour == ao_params.get('data_per_hour_30'):

        hist_centers = np.arange(day_ao_min_limit, day_ao_max_limit, ao_params['hour_bin_size'] / ao_params['data_per_hour_30'])
        hist_centers = np.append(hist_centers, hist_centers[-1] + ao_params['hour_bin_size'] / ao_params['data_per_hour_30'])

    elif data_per_hour == static_params.get('ao').get('data_per_hour_15'):

        hist_centers = np.arange(day_ao_min_limit, day_ao_max_limit, ao_params['hour_bin_size'] / ao_params['data_per_hour_15'])
        hist_centers = np.append(hist_centers, hist_centers[-1] + ao_params['hour_bin_size'] / ao_params['data_per_hour_15'])

    else:

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

    if epoch_count_so_far >= ao_params.get('data_per_hour_30') + 1:

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

            bin_counts      (np.ndarray)      : input data matrix with 21 columns
            hist_centers    (np.ndarray)      : object to log important steps of always on code flow
            data_per_hour   (np.ndarray)      : dictionary containing user's config related parameters

        Returns:

            day_ao          (np.ndarray)      : always on estimate at epoch level
        """

    static_params = hvac_static_params()
    ao_params = static_params.get('ao')

    consumption_so_far = []
    epoch_count_so_far = 0

    for index in range(len(bin_counts)):

        epoch_count_so_far = epoch_count_so_far + bin_counts[index]
        consumption_so_far.append(bin_counts[index] * hist_centers[index])

        day_ao = 0
        break_bool = False

        if bin_counts[index] >= data_per_hour:

            day_ao = hist_centers[index]
            break_bool = True

        elif bin_counts[index] < data_per_hour:

            if data_per_hour == ao_params.get('data_per_hour_30'):

                day_ao, break_bool = get_30_min_sampling_ao(epoch_count_so_far, consumption_so_far, bin_counts, index, day_ao, break_bool)

            elif data_per_hour == ao_params.get('data_per_hour_15'):

                day_ao, break_bool = get_15_min_sampling_ao(epoch_count_so_far, consumption_so_far, bin_counts, index, day_ao, break_bool)

            else:

                day_ao, break_bool = get_general_sampling_ao(epoch_count_so_far, consumption_so_far, bin_counts, index, day_ao, break_bool)


        if break_bool == True:
            break

    return day_ao


def compute_day_level_ao(input_data, logger_ao, global_config):

    """
    Function to compute ao at epoch level

    Parameters:

        input_data (np.ndarray)     : input data matrix with 21 columns
        logger_ao (logger object)    : object to log important steps of always on code flow
        global_config (dict)         : dictionary containing user's config related parameters

    Returns:

        epoch_algo_ao (np.ndarray)  : always on estimate at epoch level
        wholehouse (np.ndarray)     : consumption left at epoch level after ao is removed
        exit_status (dict)           : dictionary containing exit status and error codes for ao module
    """

    # if input data is empty, return fail-safe estimate of ao and wholehouse
    if input_data.size == 0:
        epoch_ao = np.array([])
        wholehouse = np.array([])

    # getiing epoch timestamps and raw consumptions
    epoch_data = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]
    epoch_raw_data = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # ensuring no nan value flow in baseload algo
    epoch_raw_data[np.isnan(epoch_raw_data)] = 0

    sampling_rate = global_config.get('sampling_rate')
    data_per_hour = Cgbdisagg.SEC_IN_HOUR / sampling_rate

    # initialize epoch level ao vector
    number_of_epochs = input_data.shape[0]
    epoch_ao = np.nan * scipy.ones(number_of_epochs, float)

    # getting unique month epoch stamps, to be used as specific month identifier
    unique_days = scipy.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX])
    logger_ao.info(' Getting AO values for {} unique days |'.format(len(unique_days)))

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

            except IndexError:

                day_ao = (day_ao_min_limit + day_ao_max_limit) / 2
                epoch_ao[day_idx] = day_ao

        else:

            day_ao = 0

        epoch_ao[day_idx] = day_ao

    logger_ao.info(' Describe AO after logic of lowest 1 hour consumption. sum : {}, min : {}, max : {} |'.format(np.nansum(epoch_ao), np.nanmin(epoch_ao), np.nanmax(epoch_ao)))

    logger_ao.info(' Epochs where (AO > Net) : {} |'.format(np.nansum(epoch_ao > epoch_raw_data)))
    epoch_ao[epoch_ao > epoch_raw_data] = epoch_raw_data[epoch_ao > epoch_raw_data]
    epoch_ao[epoch_ao < 0] = 0
    logger_ao.info(' Describe AO after handling (AO > net). sum : {}, min : {}, max : {} |'.format(np.nansum(epoch_ao), np.nanmin(epoch_ao), np.nanmax(epoch_ao)))

    epoch_raw_minus_ao = epoch_raw_data - epoch_ao

    exit_status = {
        'exit_code': 0,
        'error_list': [],
    }

    month_epoch, idx_2, month_idx = scipy.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_index=True, return_inverse=True)

    # aggregating always on estimates at month level for monthly estimates
    month_ao = np.bincount(month_idx, epoch_ao)
    logger_ao.info(' Aggregated always on estimates at month level for monthly estimates |')

    # concatenating month level appliance estimates, to form consumption array
    month_algo_ao = np.c_[month_epoch, month_ao]
    logger_ao.info(' Concatenated month level always on estimates, to form consumption array |')

    # concatenating the epoch timestamps adjacent to epoch level always on estimates, for output
    epoch_algo_ao = np.c_[epoch_data, epoch_ao]

    return month_algo_ao, epoch_algo_ao, epoch_raw_minus_ao, exit_status
