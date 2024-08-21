"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to log monthly consumptions
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def write_monthly_log(timed_wh_signal, disagg_mode, logger):
    """
    Parameters:
        timed_wh_signal         (np.ndarray)    : Timed water heater output
        disagg_mode             (str)           : Disagg mode
        logger                  (logger)        : Logger object to log values

    Returns:
        None:                   (None)          : No returned object
    """

    # Get unique months and the relevant indices

    unq_months, months_idx, _ = np.unique(timed_wh_signal[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                                          return_counts=True, return_inverse=True)

    # Aggregate consumption at monthly level

    monthly_consumption = np.bincount(months_idx, timed_wh_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    monthly_output = np.c_[unq_months.reshape(-1, 1), monthly_consumption.reshape(-1, 1)]

    # Log consumption according to different disagg mode

    if disagg_mode == 'historical':
        # Writing the monthly consumption to logs

        monthly_output_log = [(int(monthly_output[i, 0]), monthly_output[i, 1]) for i in range(monthly_output.shape[0])]

        logger.info('The monthly Timed WH consumption (in Wh) is : | %s', str(monthly_output_log).replace('\n', ' '))
    elif disagg_mode in ['incremental', 'mtd']:
        # Writing only the last bill cycle consumption

        monthly_output_log = [int(monthly_output[-1, 0]), monthly_output[-1, 1]]

        logger.info('The monthly Timed WH consumption (in Wh) is : | %s', str(monthly_output_log).replace('\n', ' '))
