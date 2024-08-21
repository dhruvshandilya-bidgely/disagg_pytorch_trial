"""
Author - Mayank Sharan
Date - 28/11/19
Helper functions to be used by vacation disagg wrapper. They also help in reducing complexity of the disagg wrapper
"""

# Import python packages

import copy
import numpy as np
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.write_estimate import write_estimate
from python3.config.pilot_constants import PilotConstants
from python3.disaggregation.aer.vacation.run_vacation_debug import run_vacation_debug
from python3.disaggregation.aer.vacation.init_vacation_config import init_vacation_config


def extract_pp_twh_for_vacation(disagg_output_object, pilot_id):

    """
    Extracts pool pump and timed water heater consumption form other modules to use in vacation disagg

    Parameters:
        disagg_output_object    (dict)              : Dictionary containing all outputs
        pilot_id                (int)               : The id for the pilot the user belongs to

    Returns:
        timed_disagg_output     (dict)              : Dict containing all timed appliance outputs
    """

    # Extract the pool pump output from source based on pilot id

    if pilot_id in PilotConstants.AUSTRALIA_PILOTS:

        # If running vacation module for the first time (without running pool pump)

        if not len(disagg_output_object.get('special_outputs')):

            pp_disagg_output = np.zeros_like(disagg_output_object.get('epoch_estimate').shape[0])

        else:

            # Extract pool pump output from special output for Origin

            cons_col = 1

            if disagg_output_object.get('special_outputs').get('pp_consumption') is not None:
                pp_disagg_output = copy.deepcopy(disagg_output_object.get('special_outputs').get('pp_consumption')[:, cons_col])

            else:
                pp_disagg_output = np.zeros_like(disagg_output_object.get('epoch_estimate').shape[0])

    else:

        # Extract timestamp level pool pump consumption data

        pp_out_idx = disagg_output_object.get('output_write_idx_map').get('pp')
        pp_disagg_output = copy.deepcopy(disagg_output_object.get('epoch_estimate')[:, pp_out_idx])

    # Set nans to 0 and if pp is absent set to empty array

    pp_disagg_output[np.isnan(pp_disagg_output)] = 0
    pp_cons_pts = np.sum(pp_disagg_output)

    # Extract timestamp level timed water heater consumption data

    twh_disagg_output = disagg_output_object.get("special_outputs").get("timed_water_heater")

    if twh_disagg_output is None:
        twh_disagg_output = np.zeros_like(disagg_output_object.get('epoch_estimate').shape[0])
    else:
        twh_disagg_output = twh_disagg_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Set nans to 0 and if twh is absent set to empty array

    twh_disagg_output[np.isnan(twh_disagg_output)] = 0
    twh_cons_pts = np.sum(twh_disagg_output)

    # Populate the timed disagg output dictionary

    timed_disagg_output = {
        'pp': {
            'cons': pp_disagg_output,
            'num_pts': pp_cons_pts,
        },
        'twh': {
            'cons': twh_disagg_output,
            'num_pts': twh_cons_pts,
        },
    }

    return timed_disagg_output


def get_vacation_inputs(disagg_input_object, disagg_output_object):

    """
    Extracts inputs required for vacation disagg

    Parameters:
        disagg_input_object     (dict)              : Dictionary containing all inputs
        disagg_output_object    (dict)              : Dictionary containing all outputs

    Returns:
        input_data              (np.ndarray)        : 21 column data processed to be used by vacation
        vacation_config         (dict)              : Dictionary containing config variables for vacation run
        timed_disagg_output     (dict)              : Contains timed devices disagg that have been computed
        exit_status             (dict)
    """

    # Extract 21 column input data and put nans back in place to indicate missing data

    input_data = copy.deepcopy(disagg_input_object.get('input_data'))

    is_nan_cons = disagg_input_object.get('data_quality_metrics').get('is_nan_cons')

    if is_nan_cons is not None:
        input_data[is_nan_cons, Cgbdisagg.INPUT_CONSUMPTION_IDX] = np.nan

    # Extract timezone, sampling rate, pilot id and uuid and use it to initialize vacation config

    global_config = disagg_input_object.get('config')

    uuid = global_config.get('uuid')
    pilot_id = global_config.get('pilot_id')
    sampling_rate = global_config.get('sampling_rate')

    timezone = disagg_input_object.get('home_meta_data').get('timezone')

    vacation_config = init_vacation_config(pilot_id, sampling_rate, timezone, uuid)

    # Get timed devices consumption

    timed_disagg_output = extract_pp_twh_for_vacation(disagg_output_object, pilot_id)

    # Initialize the exit status dictionary

    exit_status = {
        'exit_code': 1,
        'error_list': [],
    }

    return input_data, vacation_config, timed_disagg_output, exit_status


def write_vacation_results(disagg_input_object, disagg_output_object, debug, type_1_epoch, type_2_epoch, global_config,
                           vacation_config, logger_vacation):

    """
    If vacation has run write results to disagg output object and log information

    Parameters:
        disagg_input_object     (dict)              : Dictionary containing all inputs
        disagg_output_object    (dict)              : Dictionary containing all outputs
        debug                   (dict)              : Contains all variables needed for debugging
        type_1_epoch            (np.ndarray)        : Array marking type 1 vacation at epoch level
        type_2_epoch            (np.ndarray)        : Array marking type 2 vacation at epoch level
        global_config           (dict)              : Dictionary containing configuration for the pipeline
        vacation_config         (dict)              : Dictionary containing all configuration variables for vacation
        logger_vacation         (logger)            : Logger to be used to log information

    Returns:
        disagg_output_object    (dict)              : Dictionary containing all outputs
    """

    # Initialize constants to be used to index columns

    # For type_1_epoch and type_2_epoch

    bc_col = 0
    label_col = 1

    # For bill cycle vac count

    t1_col = 1
    t2_col = 2

    # For out bill cycles

    bc_start = 0
    bc_end = 1

    # If vacation has run write results

    if debug is not None:

        # Initialize bill cycle level vacation count

        out_bill_cycles = disagg_input_object.get('out_bill_cycles')

        bill_cycle_vac_count = np.zeros(shape=(out_bill_cycles.shape[0], 3))
        bill_cycle_vac_count[:, bc_col] = out_bill_cycles[:, bc_start]

        # For each bill cycle fill the number of days detected as vacation

        num_pts_day = int(Cgbdisagg.SEC_IN_DAY / global_config.get('sampling_rate'))

        for bc_idx in range(out_bill_cycles.shape[0]):
            ts_in_bc = np.logical_and(type_1_epoch[:, bc_col] < out_bill_cycles[bc_idx, bc_end],
                                      type_1_epoch[:, bc_col] >= out_bill_cycles[bc_idx, bc_start])

            bill_cycle_vac_count[bc_idx, t1_col] = int(round(np.sum(type_1_epoch[ts_in_bc, label_col]) / num_pts_day))
            bill_cycle_vac_count[bc_idx, t2_col] = int(round(np.sum(type_2_epoch[ts_in_bc, label_col]) / num_pts_day))

        # Log the vacation as number of days corresponding to each bill cycle

        monthly_output_log_type_1 = [(datetime.utcfromtimestamp(bill_cycle_vac_count[i, bc_col]).strftime('%b-%Y'),
                                      bill_cycle_vac_count[i, t1_col]) for i in range(bill_cycle_vac_count.shape[0])]

        monthly_output_log_type_2 = [(datetime.utcfromtimestamp(bill_cycle_vac_count[i, bc_col]).strftime('%b-%Y'),
                                      bill_cycle_vac_count[i, t2_col]) for i in range(bill_cycle_vac_count.shape[0])]

        logger_vacation.info("The monthly type 1 vacation detected (in Days) is : | %s",
                             str(monthly_output_log_type_1).replace('\n', ' '))

        logger_vacation.info("The monthly type 2 vacation detected (in Days) is : | %s",
                             str(monthly_output_log_type_2).replace('\n', ' '))

        # Initialize output indices

        vac_read_idx = 1

        vacation_out_idx = disagg_output_object.get('output_write_idx_map').get('va')
        vac_1_out_idx = vacation_out_idx[0]
        vac_2_out_idx = vacation_out_idx[1]

        # Write timestamp level estimates

        disagg_output_object = write_estimate(disagg_output_object, type_1_epoch, vac_read_idx, vac_1_out_idx, 'epoch')
        disagg_output_object = write_estimate(disagg_output_object, type_2_epoch, vac_read_idx, vac_2_out_idx, 'epoch')

        # This is put in place right now just so that we are able to run this with lighting and water heater.
        # To be removed once they adapt to proper format

        disagg_output_object['special_outputs']['vacation_periods'] = debug.get('vacation_periods')

        # Run vacation debug file so that as per parameters debug info can be dumped

        run_vacation_debug(debug, vacation_config, global_config)

    return disagg_output_object
