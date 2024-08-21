"""
Author - Nikhil Singh Chauhan
Date - 16/10/18
Handy functions for water heater module
"""

# Import python packages

import numpy as np
from copy import deepcopy
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_day_ts(ts):
    """
    Find the day timestamp from epoch timestamp

    Parameters:
        ts              (int)       : Epoch timestamp

    Returns:
        day_ts          (int)       : First timestamp of the corresponding day
    """

    # Remove the extra seconds from each day to get first timestamp of day

    day_ts = datetime.utcfromtimestamp(ts)
    day_ts = ts - (Cgbdisagg.SEC_IN_HOUR * day_ts.hour + Cgbdisagg.SEC_IN_1_MIN * day_ts.minute + day_ts.second)

    # Making sure the timestamp is an integer

    day_ts = int(day_ts)

    return day_ts


def get_month_ts(ts):
    """
    Getting month first ts from epoch

    Parameters;
        ts              (int)       : Input timestamp

    Returns:
        month_ts        (int)       : First timestamp of corresponding month
    """

    # Remove the extra seconds from each month to get first timestamp of month

    month_ts = datetime.utcfromtimestamp(ts)
    month_ts = ts - (Cgbdisagg.SEC_IN_DAY * month_ts.day + Cgbdisagg.SEC_IN_HOUR * month_ts.hour +
                     Cgbdisagg.SEC_IN_1_MIN * month_ts.minute + month_ts.second)

    # Making sure the timestamp is an integer

    month_ts = int(month_ts)

    return month_ts


def bill_cycle_to_month(in_data):
    """
    Parameters:
        in_data             (np.ndarray)        : Input 21-column matrix

    Returns:
        input_data          (np.ndarray)        : Update 21-column matrix
    """

    # Taking deepcopy of input data to keep local instance

    input_data = deepcopy(in_data)

    # Get the true month from the bill cycle timestamp

    month_ts = [*map(get_month_ts, input_data[:, Cgbdisagg.INPUT_EPOCH_IDX])]

    # Update the input data bill cycle column

    input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] = month_ts

    return input_data


def find_missing_bill_cycle_season(in_data, seasons, logger):
    """
    Parameters:
        in_data             (np.ndarray)    : Input 21-column matrix
        seasons             (np.ndarray)    : All seasons data
        logger              (logger)        : Logger object

    Returns:
        all_seasons         (np.ndarray)    : Updated all seasons data
    """

    # Taking deepcopy of input data to keep local instance

    input_data = deepcopy(in_data)
    all_seasons = deepcopy(seasons)

    # Finding the unique bill cycles in input data and temperature data

    data_bill_cycles = np.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX])
    valid_bill_cycles = np.unique(all_seasons[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX])

    # If the number of bill cycles match, none is missing and therefore return

    if len(data_bill_cycles) == len(valid_bill_cycles):
        # If no missing bill cycles in the seasons data

        return all_seasons

    # Get the missing bill cycle by checking the uncommon bill cycles

    missing_bill_cycles = [bill_cycle for bill_cycle in data_bill_cycles if bill_cycle not in valid_bill_cycles]

    logger.info('Imputing seasons for bill cycles | {}'.format(missing_bill_cycles))

    # Iterate over each missing bill cycle

    for current_bill_cycle in missing_bill_cycles:
        # Find the closest few bill cycles to the missing bill cycle

        nearest_idx = np.argsort(np.abs(valid_bill_cycles - current_bill_cycle))[:2]

        # Take mean temperature of the nearest 2 bill cycles and round to 3 digits

        current_temp = np.round(np.mean(all_seasons[nearest_idx, 1]), 3)

        # Add the missing bill cycle to all seasons data

        all_seasons = np.vstack((all_seasons, np.r_[current_bill_cycle, current_temp, 0]))

    return all_seasons


def get_other_appliance_output(disagg_output_object, debug):
    """
    Parameters:
        disagg_output_object        (dict)      : Output of disagg pipeline
        debug                       (dict)      : Output from water heater algo steps

    Returns:
        debug                       (dict)      : Output from water heater algo steps
    """

    # Retrieve the other appliance output from disagg output object
    # 'ao'              : Always On output
    # 'pp'              : Pool pump output
    # 'pp_confidence'   : Pool pump detection confidence

    other_output = {
        'ao': disagg_output_object['epoch_estimate'][:, disagg_output_object.get('output_write_idx_map').get('ao')],
        'pp': disagg_output_object.get('special_outputs').get('pp_consumption'),
        'pp_confidence': disagg_output_object.get('special_outputs').get('pp_confidence'),
        'vacation': disagg_output_object['epoch_estimate'][:, disagg_output_object.get('output_write_idx_map').get('va')]
    }

    debug['other_output'] = other_output

    # Extract water heater ground truth data from disagg output object if available

    water_heater_gt = disagg_output_object.get('special_outputs').get('gt')

    debug['waterheater_gt'] = water_heater_gt

    return debug


def get_bill_cycle_info(input_data, debug):
    """
    Parameters:
        input_data          (np.ndarray)    : Input 21-column matrix
        debug               (dict)          : Dictionary with algorithm intermediate steps

    Returns:
        debug               (dict)          : Dictionary with bill cycle info
    """

    # Extracting the bill cycle info from input data

    bill_cycle_ts, bill_cycle_idx, points_count = np.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                                                            return_counts=True, return_inverse=True)

    # Saving the bill cycle timestamps and their indices

    debug['bill_cycle_ts'] = bill_cycle_ts
    debug['bill_cycle_idx'] = bill_cycle_idx
    debug['bill_cycle_count'] = points_count

    return debug
