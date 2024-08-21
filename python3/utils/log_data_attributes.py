"""
Author - Mayank Sharan
Date - 31/07/19
Logs different aspects of the data that can be useful for analysis
"""

# Import python packages

import copy
import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def log_data_attributes(pipeline_input_object, logger_pass):

    """
    Parameters:
        pipeline_input_object (dict)              : Dictionary containing all inputs
        logger_pass         (dict)              : Contains information needed to perform logging
    """

    # Initialize the logger

    logger_base = logger_pass.get('base_logger').getChild('log_data_attributes')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Copy the input data to avoid any modification

    input_data = copy.deepcopy(pipeline_input_object.get('input_data'))

    # Log number of days of data available

    logger.info('Number of days in the data are | %d', len(np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX])))

    # Log bill cycles to write output for

    num_bill_cycles = pipeline_input_object.get('out_bill_cycles').shape[0]
    logger.info('Number of billing cycles to write results for are | %d - %s', num_bill_cycles,
                str(pipeline_input_object.get('out_bill_cycles')).replace('\n', ' '))

    # Log number of data points in each bill cycle required for analysis purposes

    bill_cycles, inv_idx, pts_per_bill_cycle = np.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                                                         return_inverse=True, return_counts=True)

    data_points_log = [(int(bill_cycles[i]), pts_per_bill_cycle[i]) for i in range(bill_cycles.shape[0])]
    logger.info('Number of data points by billing cycle is | %s', str(data_points_log).replace('\n', ' '))

    # Log total consumption by billing cycle

    bc_consumption = np.bincount(inv_idx, weights=input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    total_cons_log = [(int(bill_cycles[i]), bc_consumption[i]) for i in range(bill_cycles.shape[0])]

    logger.info('Total Consumption (in Wh) by billing cycle are | %s', str(total_cons_log).replace('\n', ' '))

    # Log data check info so that we can reference in future to ensure data is same

    mean_values = np.nanmean(input_data, axis=0)

    nan_bool_matrix = np.isnan(input_data)
    count_columns = np.sum(~nan_bool_matrix, axis=0)

    check_ptiles = [0, 25, 50, 75, 100]
    check_percentile_values = np.round(np.nanpercentile(input_data, q=check_ptiles, axis=0), 5)

    # For each column of input_data log the attributes computed

    logger.info('Checks for columns heading | count, mean, %dth ptile, %dth ptile, %dth ptile, %dth ptile, %dth ptile',
                check_ptiles[0], check_ptiles[1], check_ptiles[2], check_ptiles[3], check_ptiles[4])

    for col_idx in range(input_data.shape[1]):

        logger.info('Checks for column %d | %d, %.3f, %.1f, %.1f, %.1f, %.1f, %.1f', col_idx, count_columns[col_idx],
                    mean_values[col_idx], check_percentile_values[0, col_idx], check_percentile_values[1, col_idx],
                    check_percentile_values[2, col_idx], check_percentile_values[3, col_idx],
                    check_percentile_values[4, col_idx])
