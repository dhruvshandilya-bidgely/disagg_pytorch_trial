"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Post filtering checks for energy consistency
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.maths_utils.maths_utils import rotating_sum


def bill_cycle_filter(input_box_data, box_energy_idx, wh_config, debug, logger_base):
    """
    Checking if a bill cycle has an abnormal fraction or consumption

    Parameters:
        input_box_data      (np.ndarray)    : Input 21-column box data
        box_energy_idx      (np.ndarray)    : Box energy indices
        wh_config           (dict)          : Config params
        debug               (dict)          : Algorithm intermediate steps output
        logger_base         (dict)          : Logger object

    Returns:
        box_data            (np.ndarray)    : Updated box data
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('post_filtering_checks')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Columns for bill cycle fractions

    bill_cycle_col = 0
    raw_roll_ratio = 1
    total_consumption = 2
    final_bill_cycle_tag = 3

    # Get limits from config

    fraction_limits = wh_config.get('fraction_limits')
    required_bc_count = wh_config.get('required_bc_count')
    consumption_limits = wh_config.get('consumption_limits')

    # Maximum hourly fraction is defined as 1

    max_fraction = 1

    # Taking deepcopy of input data to keep local instances

    box_data = deepcopy(input_box_data)

    # Energy diff of box energy

    box_idx_diff = np.diff(np.r_[0, box_energy_idx, 0])

    # Find edge indices for the corresponding water heater type (start / end)

    if debug['wh_type'] == 'start':
        box_edge_idx = (box_idx_diff[:-1] > 0)
    else:
        box_edge_idx = (box_idx_diff[1:] < 0)

    # Get the energy fractions

    bin_offset = 0.5

    edges = np.arange(0, debug['max_hod'] + 2) - bin_offset
    factor = debug['time_factor']

    # Take unique bill cycle timestamps and indices

    bill_cycle_ts, bill_cycle_idx = np.unique(box_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_inverse=True)

    # Stack fractions with bill cycle timestamps

    bill_cycle_fractions = np.hstack([bill_cycle_ts.reshape(-1, 1), np.zeros((len(bill_cycle_ts), 3))])

    # Iterate over each bill cycle

    for i, bill_cycle in enumerate(bill_cycle_ts):
        # Temp data of current bill cycle

        temp_box_data = box_data[box_data[:, bill_cycle_col] == bill_cycle]
        temp_box_edge_idx = box_edge_idx[np.where(bill_cycle_idx == i)[0]]

        # Number of days in current bill cycle

        temp_num_days = len(np.unique(temp_box_data[:, Cgbdisagg.INPUT_DAY_IDX]))

        # Energy fractions in the current bill cycle

        temp_hod_count, _ = np.histogram(temp_box_data[temp_box_edge_idx, Cgbdisagg.INPUT_HOD_IDX], bins=edges)
        temp_hod_count = temp_hod_count / temp_num_days
        max_raw = np.max(np.fmin(deepcopy(temp_hod_count), max_fraction))

        # Calculating roll fractions from raw fractions

        temp_hod_count = rotating_sum(temp_hod_count, factor)
        temp_hod_count = np.fmin(temp_hod_count, max_fraction)
        max_roll = np.max(temp_hod_count)

        # Appending values of raw / roll with bill cycle info

        bill_cycle_fractions[i, raw_roll_ratio] = max_raw / max_roll
        bill_cycle_fractions[i, total_consumption] = np.sum(temp_box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # Extract fractions and consumptions to compare

    fractions = bill_cycle_fractions[:, raw_roll_ratio]
    consumptions = bill_cycle_fractions[:, total_consumption]

    # Calculate the maximum allowed fraction and consumption

    max_fraction = np.mean(np.sort(fractions)[::-1][:np.min([required_bc_count, len(bill_cycle_ts)])])
    max_consumption = np.mean(np.sort(consumptions)[::-1][:np.min([required_bc_count, len(bill_cycle_ts)])])

    # Mark bill cycles with abnormally low fraction

    low_fractions = (fractions < np.max([fraction_limits[0] * max_fraction, wh_config['raw_roll_threshold']])) \
                    & (consumptions < (consumption_limits[2] * max_consumption))

    # Mark bill cycles with abnormally low consumption

    low_consumptions = consumptions < (consumption_limits[0] * max_consumption)

    # Get bill cycles which violate all the conditions

    invalid_bill_cycles = (fractions < (fraction_limits[1] * max_fraction)) \
                          & (consumptions < (consumption_limits[1] * max_consumption))

    invalid_bill_cycles = invalid_bill_cycles | low_fractions  | low_consumptions

    # Violated bill cycles' consumption is made zero

    bill_cycle_fractions[:, final_bill_cycle_tag] = invalid_bill_cycles.astype(int)

    # Removing the invalid bill cycle consumption

    for i, row in enumerate(bill_cycle_fractions):
        if row[final_bill_cycle_tag] == 1:
            # If the bill cycle is abnormal, mask consumption as zero

            box_data[box_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == row[bill_cycle_col],
                     Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    logger.info('Bill cycles removed are | %s', bill_cycle_ts[invalid_bill_cycles].astype(int))

    return box_data


def fit_winter_energy(timed_box, wtr_idx, input_data):
    """
    Adding winter energy back with the rest of the data

    Parameters:
        timed_box           (np.ndarray)    : Input 21-column box data
        wtr_idx             (np.ndarray)    : Winter indices
        input_data          (np.ndarray)    : Input 21-column raw data

    Returns:
        final_timed_box     (np.ndarray)    : Updated final timed water heater output
    """

    # Initializing deepcopy of input data to keep local instances

    final_timed_box = deepcopy(input_data)
    final_timed_box[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    # Replace winter indices consumption with winter detection module output

    final_timed_box[wtr_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = timed_box[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    return final_timed_box
