"""
Author - Sahana M
Date - 4/3/2020
Wrapper file for ref
"""

# Import python packages

import logging
import numpy as np

# import functions from within the project

from python3.itemization.aer.ref.ref_module import ref_hybrid_module
from python3.utils.write_estimate import write_estimate


def get_monthly_estimate(month_ts, data_est):

    """ Return monthly_Estimate

    This function generates estimated usage per month

    Input:
      month_ts (double matrix)    : 2d matrix with month timestamps
      data_est (double matrix)    : 2d matrix containing lighting estimates

    Output:
      monthly_Estimate (double matrix) : Matrix with month timestamp and estimate
    """

    col_size = month_ts.size
    month_ts_col = np.reshape(month_ts, [col_size, 1], order='F').copy()
    data_est_col = np.reshape(data_est, [col_size, 1], order='F').copy()

    val_indices = ~np.isnan(month_ts_col)

    month_ts_col = month_ts_col[val_indices]
    data_est_col = data_est_col[val_indices]

    ts, _, idx = np.unique(month_ts_col, return_index=True, return_inverse=True)

    monthly_estimate = np.bincount(idx, weights=data_est_col)
    dop = np.zeros(shape=ts.shape)
    monthly_estimate = np.c_[ts, monthly_estimate, dop, dop, dop]

    return monthly_estimate


def ref_hybrid_wrapper(item_input_object, item_output_object, disagg_output_object):

    """
    Wrapper file for ref
    Args:
        item_input_object             (dict)      : Dictionary containing all hybrid inputs
        item_output_object            (dict)      : Dictionary containing all hybrid outputs
        disagg_output_object            (dict)      : Dictionary containing all disagg outputs

    Returns:
        itemisation_input_object         (dict)      : Dictionary containing all inputs
        itemisation_output_object        (dict)      : Dictionary containing all outputs
    """

    # Initiate logger for the ref module

    logger_ref_base = item_input_object.get('logger').getChild('ref_hybrid_wrapper')
    logger_ref = logging.LoggerAdapter(logger_ref_base, item_input_object.get('logging_dict'))
    logger_ref_pass = {
        'logger_base': logger_ref_base,
        'logging_dict': item_input_object.get('logging_dict'),
    }

    # Initialise arguments to be given to the ref module

    global_config = item_input_object.get('config')

    # Get disagg_mode and run_mode from the input object

    run_mode = global_config.get('run_mode')

    # Decide on which mode to run based on the run mode

    if run_mode == 'prod' or run_mode == 'custom' or run_mode == None:
        # Calling the ref module

        item_input_object, item_output_object, ref_estimate, hybrid_ref, day_estimate = ref_hybrid_module(item_input_object, item_output_object, logger_ref_pass)

        ref_out_idx = item_output_object.get('output_write_idx_map').get('ref')

        item_output_object["ref"] = dict()
        item_output_object["ref"]["hybrid_ref"] = hybrid_ref

        ref_read_idx = 1

        month_ts =  item_input_object.get("item_input_params").get("month_ts")

        monthly_ref = get_monthly_estimate(month_ts, day_estimate)

        item_output_object = write_estimate(item_output_object, monthly_ref, ref_read_idx, ref_out_idx, 'bill_cycle')

        item_output_object = write_estimate(item_output_object, ref_estimate , ref_read_idx, ref_out_idx, 'epoch')

    else:

        logger_ref.error('Unrecognized disagg mode %s |', global_config.get('disagg_mode'))

    return item_input_object, item_output_object, disagg_output_object
