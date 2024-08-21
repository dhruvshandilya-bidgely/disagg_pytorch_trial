"""
Author - Nikhil Singh Chauhan
Date - 15-May-2020
Module with functions to validate monthly / epoch level EV output
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def check_month_output(monthly_output, bill_cycle_est, logger_ev):

    """
    Parameters:
        monthly_output          (np.ndarray)        : Bill cycle output for water heater
        bill_cycle_est          (np.ndarray)        : Bill cycle output for all appliances
        logger_ev               (logger)			: Logger object

    Returns:
        monthly_output          (np.ndarray)        : Bill cycle output for water heater
    """

    if monthly_output.shape[0] > 0:
        logger_ev.info('Monthly estimate output received from algo successfully | ')
    else:
        logger_ev.warning('Monthly estimate output NOT received from algo, creating dummy data | ')

        # Creating dummy data per billing cycle

        monthly_output = np.zeros((bill_cycle_est.shape[0], 5))
        monthly_output[:, 0] = bill_cycle_est[:, 0]

    return monthly_output


def check_epoch_output(debug, epoch_est, logger_ev):

    """
    Parameters:
        debug                   (dict)              : Dictionary containing all inputs
        epoch_est               (np.ndarray)        : Dictionary containing all outputs
        logger_ev               (logger)			: Logger object

    Returns:
        ts_1d                   (np.ndarray)        : Dictionary containing all inputs
        ev_1d                   (np.ndarray)        : Dictionary containing all outputs
    """

    # noinspection PyBroadException
    try:
        # Epoch timestamps with water heater output
        ts_1d = debug.get('final_ev_signal')[:, Cgbdisagg.INPUT_EPOCH_IDX]

        # Water heater output values at each epoch
        ev_1d = debug.get('final_ev_signal')[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

        logger_ev.info('Timestamp and Estimate received from debug object | ')
    except (KeyError, IndexError, TypeError):
        logger_ev.warning('Epoch estimate NOT received from algo, creating dummy data | ')

        # Creating dummy data
        ts_1d = np.zeros(epoch_est.shape[0])
        ev_1d = np.zeros(epoch_est.shape[0])

    return ts_1d, ev_1d
