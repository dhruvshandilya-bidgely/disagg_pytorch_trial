"""
Author - Sahana M
Date - 2/3/2021
Computes wh_potential starting and ending days along with buffer days
"""

# Import python packages
import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.itemization.aer.water_heater.functions.math_utils import find_seq


def get_no_wh_potential_days(no_wh_seq, seq_days):
    """
    This function is used to identify the starting and ending index of days with no WH potential
    Args:
        no_wh_seq       (np.ndarray)        : Contains all the start & end index of days with 0 wh_potential
        seq_days                (np.ndarray)        : Contains all the start & end index of days with wh_potential
    Returns:
        start_idx       (int)               : Starting index of the No wh potential array
        end_idx         (int)               : Ending index of the No wh potential array
    """
    # If wh potential present throughout the year

    if not len(no_wh_seq):
        start_idx = 0
        end_idx = 0

    else:

        # If wh potential present during starting and ending of the input data join them together

        wh_pot_starting = seq_days[0, 0] == 0
        wh_pot_ending = seq_days[-1, 0] == 0
        min_chunks = seq_days.shape[0] >= 2

        if wh_pot_starting & wh_pot_ending & min_chunks:
            no_wh_seq = np.r_[no_wh_seq[1:-1, :],
                              [[0,
                                no_wh_seq[-1, 1],
                                no_wh_seq[0, 2],
                                (no_wh_seq[-1, 3] + no_wh_seq[0, 3])]]]

        # get the longest no wh potential days indexes

        longest_no_wh_seq_idx = np.argmax(no_wh_seq[:, 3])
        start_idx = no_wh_seq[longest_no_wh_seq_idx, 1]
        end_idx = no_wh_seq[longest_no_wh_seq_idx, 2]

    return start_idx, end_idx


def check_common_buffer_days(transition_edge_indexes, buffer_edge_indexes):
    """
    This function is used to check if the Transition 1 and Transition 2 days are common, if so update Transition 2 indexes
    Args:
        transition_edge_indexes     (list)          : List of indexes containing transition days indexes
        buffer_edge_indexes         (list)          : List of indexes containing the surrounding days of buffer days

    Returns:
        trn_start_2                 (int)           : Updated transition 2 start index
        trn_end_2                   (int)           : Updated transition 2 end index
    """

    start = buffer_edge_indexes[0]
    mid = buffer_edge_indexes[1]
    end = buffer_edge_indexes[2]
    trn_start_1 = transition_edge_indexes[0]
    trn_start_2 = transition_edge_indexes[2]
    trn_end_1 = transition_edge_indexes[1]
    trn_end_2 = transition_edge_indexes[3]

    # If common buffer days then expand the buffer days

    common_buffer_start = trn_start_1 == trn_start_2
    common_buffer_end = trn_end_1 == trn_end_2
    trn_not_at_start = trn_start_1 != 0
    trn_not_at_end = trn_end_1 != 0

    if trn_not_at_start and trn_not_at_end and common_buffer_start and common_buffer_end:
        trn_start_2 = max(mid - 7, start)
        trn_end_2 = min(mid + 7, end)

    return trn_start_2, trn_end_2


def get_buffer_days(no_wh_seq, seq_days, wh_present_idx, cooling_pot_thr, cooling_potential, seasonal_wh_config, logger_pass):

    """
    Returns the list of starting and ending index of wh_potential along with buffer days index
    Parameters:
        no_wh_seq               (np.ndarray)        : Contains all the start & end index of days with 0 wh_potential
        seq_days                (np.ndarray)        : Contains all the start & end index of days with wh_potential
        wh_present_idx          (np.array)          : Boolean array where True indicates a day with + wh_potential
        cooling_pot_thr         (int)               : Cooling potential threshold
        cooling_potential       (np.ndarray)        : Contains cooling potential at epoch level
        seasonal_wh_config      (dict)              : Dictionary containing all needed configuration variables
        logger_pass             (Logger)            : Logger object

    Returns:
        padding_days            (list)              : wh_potential starting & ending index days along with buffer
                                                      days index
    """

    # Initialize logger

    logger_base = logger_pass.get('base_logger').getChild('get_buffer_days')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    wh_present_indexes = deepcopy(wh_present_idx)
    min_buffer_days = seasonal_wh_config.get('config').get('min_buffer_days')

    # Get the starting and ending index of days with No wh potential

    start_idx, end_idx = get_no_wh_potential_days(no_wh_seq, seq_days)

    logger.info('The start and end idx for No WH potential days are | {}, {}'.format(start_idx, end_idx))

    # Get the buffer days which include cooling potential

    if cooling_pot_thr != 'NA':

        # get the cooling days

        cooling_days = cooling_potential > 0

        # Get the common days for cooling and wh potential

        cool_days = cooling_days | wh_present_indexes

        # Get the days with presence of only cooling

        trn_days_seq = find_seq(cool_days)
        trn_days_seq = trn_days_seq[trn_days_seq[:, 0] == 0, :]

        # Ending part of the wh potential

        try:

            start = trn_days_seq[np.where(trn_days_seq[:, 2] == end_idx), 1][0][0]
            end = trn_days_seq[np.where(trn_days_seq[:, 2] == end_idx), 2][0][0]

            # get the middle most buffer days

            mid = int(start + (end - start) / 2)

            # make sure that there are at least 8 days in total (4 days from middle on each side)

            if end - mid >= (min_buffer_days / 2):
                trn_end_1 = mid + (min_buffer_days / 2)
                trn_start_1 = mid - (min_buffer_days / 2)

            # if between 4-8 days make sure not less than 4 days in total (2 days from middle on each side)

            elif end - mid >= (min_buffer_days / 4):
                trn_end_1 = mid + (end - mid)
                trn_start_1 = mid - (end - mid)

            else:
                trn_start_1 = 0
                trn_end_1 = 0

        except (IndexError, ValueError, TypeError):
            trn_start_1 = 0
            trn_end_1 = 0
            start = mid = end = 0
            logger.debug('Assigning default Transition 1 start and end indexes | ')

        trn_start_1 = max(0, trn_start_1)
        trn_end_1 = min(len(wh_present_idx), trn_end_1)
        logger.debug('Transition 1 start and end indexes | {}, {}'.format(trn_start_1, trn_end_1))

        # Starting part of the wh_potential

        try:

            start = trn_days_seq[np.where(trn_days_seq[:, 1] == start_idx), 1][0][0]
            end = trn_days_seq[np.where(trn_days_seq[:, 1] == start_idx), 2][0][0]

            # get the middle most buffer days

            mid = int(start + (end - start) / 2)

            # make sure that there are at least 8 days in total (4 days from middle on each side)

            if end - mid >= (min_buffer_days / 2):
                trn_end_2 = mid + (min_buffer_days / 2)
                trn_start_2 = mid - (min_buffer_days / 2)

            # if between 4-8 days make sure not less than 4 days in total (2 days from middle on each side)

            elif end - mid >= 2:
                trn_end_2 = mid + (end - mid)
                trn_start_2 = mid - (end - mid)

            else:
                trn_start_2 = 0
                trn_end_2 = 0

        except (IndexError, ValueError, TypeError):
            trn_start_2 = 0
            trn_end_2 = 0
            start = mid = end = 0
            logger.debug('Assigning default Transition 2 start and end indexes | ')

        trn_start_2 = max(0, trn_start_2)
        trn_end_2 = min(len(wh_present_idx), trn_end_2)
        logger.debug('Transition 2 start and end indexes | {}, {}'.format(trn_start_2, trn_end_2))

        # Check for common buffer days and if so update the transition 2 indexes

        buffer_edge_indexes = [start, mid, end]
        transition_edge_indexes = [trn_start_1, trn_end_1, trn_start_2, trn_end_2]

        trn_start_2, trn_end_2 = check_common_buffer_days(transition_edge_indexes, buffer_edge_indexes)

    else:
        trn_start_1 = 0
        trn_end_1 = 0
        trn_start_2 = 0
        trn_end_2 = 0
        logger.debug('Cooling potential not found hence no Buffer days identified | ')

        # Combine them all into a list

    logger.info('Transition buffer days 1 start and end indexes are | {}, {}'.format(trn_start_1, trn_end_1))
    logger.info('Transition buffer days 2 start and end indexes are | {}, {}'.format(trn_start_2, trn_end_2))

    padding_days = [int(start_idx), int(end_idx), int(trn_start_1), int(trn_end_1), int(trn_start_2), int(trn_end_2)]

    return padding_days
