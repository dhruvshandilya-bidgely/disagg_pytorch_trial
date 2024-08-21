"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Master file for itemization pipeline
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff


from python3.itemization.aer.raw_energy_itemization.inference_engine.run_inference_engine import get_app_cons_range

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.get_final_consumption import get_final_consumption

from python3.itemization.aer.raw_energy_itemization.appliance_potential.get_appliance_potential import get_appliance_potential

from python3.itemization.aer.raw_energy_itemization.residual_analysis.run_residual_analysis_modules import run_residual_analyis_modules


def run_itemization_modules(item_input_object, item_output_object, logger_pass):

    """
    Call sub modules of itemization pipeline

    Parameters:
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        logger_pass                 (logger)    : logger object

    Returns:
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
    """

    logger_base = logger_pass.get('logger_base').getChild('run_itemization_modules')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t0 = datetime.now()

    item_output_object = run_residual_analyis_modules(item_input_object, item_output_object, logger, logger_pass)

    t1 = datetime.now()

    logger.info("Residual analysis of the leftover consumption took %.3f s | ", get_time_diff(t0, t1))

    # Calculating appliance potential and confidence at timestamp level

    item_output_object = get_appliance_potential(item_input_object, item_output_object, logger_pass)

    t2 = datetime.now()

    logger.info("Calculation of ts level confidence and potential of all appliances took %.3f s | ", get_time_diff(t0, t1))

    # Calculating appliance consumption ranges at timestamp level

    item_output_object = get_app_cons_range(item_input_object, item_output_object, logger_pass)

    t3 = datetime.now()

    logger.info("Calculation of ts level ranges of all appliances took %.3f s | ", get_time_diff(t0, t1))

    # Final itemization modules to perform distribute raw energy into appliances

    item_output_object = get_final_consumption(item_input_object, item_output_object, logger_pass)

    t4 = datetime.now()

    logger.info("Final raw energy itemization completed %.3f s | ", get_time_diff(t0, t1))

    item_output_object["run_time"][2] = get_time_diff(t0, t1)
    item_output_object["run_time"][3] = get_time_diff(t1, t2)
    item_output_object["run_time"][4] = get_time_diff(t2, t3)
    item_output_object["run_time"][5] = get_time_diff(t3, t4)

    app_list = item_output_object['final_itemization']['appliance_list']
    tou_itemization = item_output_object['final_itemization']['tou_itemization']

    month_ts = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX]

    for app in app_list:

        if app in ["total", "va1", "va2"]:
            continue

        cons = tou_itemization[np.where(app_list == app)[0][0]]

        monthly_cons = get_monthly_estimate(month_ts, cons)

        monthly_cons = monthly_cons[monthly_cons[:, 0] > 0]

        monthly_output_log = [(datetime.utcfromtimestamp(monthly_cons[i, 0]).strftime('%b-%Y'), round(monthly_cons[i, 1]/1000, 3))
                              for i in range(monthly_cons.shape[0])]

        logger.info('The monthly itemization output (in Wh) for %s is : | %s',
                    app, str(monthly_output_log).replace('\n', ' '))

    return item_input_object, item_output_object


def get_monthly_estimate(month_ts, data_est):

    """ Return monthly_Estimate

    GETMONTHLYESTIMATE generates estimated usage per month

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
