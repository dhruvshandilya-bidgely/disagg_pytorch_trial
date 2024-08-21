"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update AO consumption ranges using inference rules
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.itemization_utils import get_index_array
from python3.itemization.aer.raw_energy_itemization.utils import update_hybrid_object

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config


def get_ao_inference(app_index, item_input_object, item_output_object, date_list, logger_pass):

    """
    Update AO consumption ranges using inference rules

    Parameters:
        app_index                   (int)       : Index of app in the appliance list
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        date_list                   (np.ndarray): list of target dates for heatmap dumping
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)      : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    config = get_inf_config().get("ao")

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_ao_inference')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Fetching required inputs

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    disagg_cons = item_output_object.get("inference_engine_dict").get("output_data")[app_index, :, :]
    mid_cons = item_output_object.get("inference_engine_dict").get("appliance_mid_values")[app_index, :, :]
    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    mid_cons = np.percentile(mid_cons, 80, axis=1)[:, None]
    mid_cons = np.minimum(mid_cons, input_data)

    mid_cons_copy = copy.deepcopy(mid_cons)

    ############################ RULE 1 - Month to month stability in AO ###########################################

    bill_cycle = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]

    unique_bc = np.unique(bill_cycle)

    weights_array = config.get("bc_weightage")

    length = len(weights_array)

    if len(unique_bc) >= length:

        vacation_count = np.zeros(len(unique_bc))
        bill_cycle_cons = np.zeros(len(unique_bc))

        for bill_cycle_index in range(len(unique_bc)):
            bill_cycle_cons[bill_cycle_index] = np.percentile(disagg_cons[bill_cycle == unique_bc[bill_cycle_index]], 80)

        for bill_cycle_index in range(len(unique_bc)):
            vacation_count[bill_cycle_index] = np.sum(vacation[bill_cycle == unique_bc[bill_cycle_index]])

        bill_cycle_cons_copy = copy.deepcopy(bill_cycle_cons)

        for bill_cycle_index in range(len(unique_bc)):
            if not (vacation_count[bill_cycle_index] > config.get("vacation_count_limit")) and \
                    not ((bill_cycle_cons[bill_cycle_index] == bill_cycle_cons[(bill_cycle_index+1) % len(unique_bc)]) and
                         (bill_cycle_cons[bill_cycle_index] == bill_cycle_cons[(bill_cycle_index-1) % len(unique_bc)])):

                logger.debug("Updating consumption for bill cycle index %d", bill_cycle_index)

                index_array = get_index_array(bill_cycle_index - int(length/2), bill_cycle_index + int(length/2), len(unique_bc))
                bc_array = bill_cycle_cons[index_array]

                bc_array[vacation_count[index_array] > config.get("vacation_count_limit")] = bc_array[int(length/2)]

                # Wont be changing disagg consumption if bc level output is 0

                if bc_array[2] == 0:
                    continue

                bill_cycle_cons_copy[bill_cycle_index] = (np.multiply(weights_array, bc_array)).sum()

    # Sanity checks

    min_cons = np.minimum(disagg_cons, mid_cons_copy)
    max_cons = np.maximum(disagg_cons, mid_cons_copy)

    min_cons[np.logical_and(disagg_cons > 0, min_cons == 0)] = disagg_cons[np.logical_and(disagg_cons > 0, min_cons == 0)]
    max_cons[np.logical_and(disagg_cons > 0, max_cons == 0)] = disagg_cons[np.logical_and(disagg_cons > 0, max_cons == 0)]
    mid_cons[np.logical_and(disagg_cons > 0, mid_cons == 0)] = disagg_cons[np.logical_and(disagg_cons > 0, mid_cons == 0)]

    min_cons = np.maximum(disagg_cons, min_cons)
    max_cons = np.maximum(disagg_cons, max_cons)

    min_cons = np.minimum(min_cons, input_data)
    max_cons = np.minimum(max_cons, input_data)

    max_cons = np.nan_to_num(max_cons)
    min_cons = np.nan_to_num(min_cons)

    # Updating the values in the original dictionary

    item_output_object = update_hybrid_object(app_index, item_output_object, mid_cons_copy, max_cons, min_cons)

    t_end = datetime.now()

    logger.info("AO inference calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object
