"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Call the Electric Vehicle disaggregation module and write output
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants

from python3.utils.write_estimate import write_estimate

from python3.utils.time.get_time_diff import get_time_diff

from python3.initialisation.load_files.load_ev_models import get_ev_models

from python3.disaggregation.aer.ev.ev_module import ev_module

from python3.disaggregation.aer.ev.functions.output_validation import check_epoch_output
from python3.disaggregation.aer.ev.functions.output_validation import check_month_output

from python3.disaggregation.aer.ev.functions.ev_hsm_utils import get_ev_hsm
from python3.disaggregation.aer.ev.functions.initialize_debug import initialize_debug
from python3.disaggregation.aer.ev.functions.get_ev_app_profile import get_ev_app_profile

from python3.disaggregation.aer.ev.functions.ev_utils import get_other_appliance_output

from python3.disaggregation.aer.ev.init_ev_params import init_ev_params

from python3.disaggregation.aer.ev.functions.get_ev_user_profile import get_ev_profile
from python3.disaggregation.aer.ev.functions.ev_utils import remove_ev_cnn_models


def ev_disagg_wrapper(disagg_input_object, disagg_output_object):
    """
    Parameters:
        disagg_input_object     (dict)              : Dictionary containing all inputs
        disagg_output_object    (dict)              : Dictionary containing all outputs

    Returns:
        disagg_input_object     (dict)              : Dictionary containing all inputs
        disagg_output_object    (dict)              : Dictionary containing all outputs
    """

    # Initiate logger for the electric vehicle module

    logger_ev_base = disagg_input_object.get('logger').getChild('ev_disagg_wrapper')
    logger_ev_pass = {'logger': logger_ev_base,
                      'logging_dict': disagg_input_object.get('logging_dict')}

    # Function specific logger

    logger_ev = logging.LoggerAdapter(logger_ev_base, disagg_input_object.get('logging_dict'))

    # Variable to store the status of algo run success/failure and the known errors
    # 'exit_code' : status of code run (1: Success, 0: Failure with known errors, -1: Failure with unknown errors)
    # 'error_list': List of handles errors encountered

    exit_status = {
        'exit_code': 0,
        'error_list': []
    }

    # Starting the algorithm time counter for EV module

    time_ev_start = datetime.now()

    # List to store the errors throughout the algorithm run

    error_list = []

    # Reading global configuration from disagg_input_object

    global_config = disagg_input_object.get('config')

    # Initialize the parameters required for Water Heater algorithm

    ev_config = init_ev_params(disagg_input_object, global_config)

    # Bill cycles for which output is to written (first column)

    out_bill_cycles = disagg_input_object.get('out_bill_cycles')[:, 0]

    # Get HSM (if present) and check for invalid hsm in this mode

    hsm_in, hsm_fail = get_ev_hsm(disagg_input_object, global_config, logger_ev_pass)

    # Read the EV models from input object

    models, detection_model_present = get_ev_models(disagg_input_object, logger_ev_pass)

    # Initialize debug object

    debug = initialize_debug(global_config, hsm_in, out_bill_cycles, models)

    # Reading input data from disagg_input_object

    input_data = deepcopy(disagg_input_object.get('input_data'))

    # Replacing the NaNs that were made zero while pre-processing

    is_nan_cons = disagg_input_object['data_quality_metrics']['is_nan_cons']
    input_data[is_nan_cons, Cgbdisagg.INPUT_CONSUMPTION_IDX] = np.nan

    # Reading appliance profile of the user for EV present

    ev_present, ev_app_profile_yes = get_ev_app_profile(disagg_input_object, logger_ev)

    # Putting together other appliance output and ground truth if needed in EV

    debug = get_other_appliance_output(disagg_output_object, debug)

    debug['ev_app_profile_yes'] = ev_app_profile_yes
    # Get disagg_mode and run_mode from the input object

    run_mode = global_config.get('run_mode')
    disagg_mode = global_config.get('disagg_mode')
    enable_ev_by_default = global_config.get('enable_ev_by_default')

    logger_ev.info('EV Present flag | {} '.format(ev_present))
    logger_ev.info('EV App profile yes flag | {} '.format(ev_app_profile_yes))
    logger_ev.info('Payload EV Enabling flag is | {} '.format(enable_ev_by_default))

    if not ev_app_profile_yes and not enable_ev_by_default:

        # Not running the ev module if EV app profile doesn't say yes

        logger_ev.warning('Not running EV module because App profile status is {} & Payload EV Enabling flag is {} '
                          '| '.format(ev_app_profile_yes, enable_ev_by_default))

    elif not detection_model_present:
        # Not running the ev module if detection model files are not present

        logger_ev.warning('Not running EV module because detection model files are not present | ')

    # Check if EV module to be run on this mode based on hsm and config

    elif (run_mode == 'prod' or run_mode == 'custom') and (hsm_fail is False):
        # If run mode valid

        if (disagg_mode == 'historical') or (disagg_mode == 'incremental'):
            # Historical / Incremental mode

            # Calling the main EV disagg module

            monthly_output, debug, hsm, exit_status = ev_module(input_data, ev_config, global_config, ev_present,
                                                                debug, exit_status, error_list, logger_ev_pass)

            # Saving the new HSM to the disagg_output_object

            disagg_output_object['created_hsm']['ev'] = hsm

        elif disagg_mode == 'mtd':
            # MTD run mode

            # Calling the main EV disagg module

            monthly_output, debug, hsm, exit_status = ev_module(input_data, ev_config, global_config, ev_present,
                                                                debug, exit_status, error_list, logger_ev_pass)
        else:
            # If invalid run mode, define the default empty output

            debug = {}
            monthly_output = np.array([])

            logger_ev.error('Unrecognized disagg mode {} | '.format(disagg_mode))
    else:
        # If run mode valid but HSM not found for this mode

        debug = {}
        monthly_output = np.array([])

        logger_ev.info('EV algorithm did not run | ')

        logger_ev.info('Run mode | {}'.format(run_mode))
        logger_ev.info('HSM fail status | {}'.format(hsm_fail))

    # If valid HSM is there, write epoch/bill_cycle output to the disagg_output_object

    if hsm_fail:
        # If EV module did not run, log the values

        logger_ev.warning('EV module did not run because %s mode needs HSM and it is missing | ',
                          disagg_mode)

    elif not ev_app_profile_yes and not enable_ev_by_default:
        logger_ev.warning('Not running EV module because App profile status is {} & Payload EV Enabling flag is {} '
                          '| '.format(ev_app_profile_yes, enable_ev_by_default))

    elif not detection_model_present:
        logger_ev.warning('EV module did not run because detection model files are not present | ')

    else:
        # Read index from disagg output object for which output is to written

        ev_out_idx = disagg_output_object.get('output_write_idx_map').get('ev')
        read_col_idx = 1

        # Read the table containing billing cycle estimate for each appliance

        bill_cycle_est = disagg_output_object.get('bill_cycle_estimate')

        # Check if monthly estimate received from algo, else create dummy data

        monthly_output = check_month_output(monthly_output, bill_cycle_est, logger_ev)

        # Log the monthly output

        monthly_output_log = [(datetime.utcfromtimestamp(monthly_output[i, 0]).strftime('%b-%Y'), monthly_output[i, 1])
                              for i in range(monthly_output.shape[0])]

        logger_ev.info('The monthly EV consumption (in Wh) is : | %s', str(monthly_output_log).replace('\n', ' '))

        # Write the EV estimate at the appropriate column and update the 'bill_cycle_estimate' in the
        # disagg output object

        disagg_output_object = write_estimate(disagg_output_object, monthly_output, read_col_idx, ev_out_idx,
                                              'bill_cycle')

        # Pick timestamps for which output is to written

        epoch_est = disagg_output_object.get('epoch_estimate')

        # Check if the epoch level received from the algo, else create dummy data

        ts_1d, ev_1d = check_epoch_output(debug, epoch_est, logger_ev)

        # Write the estimate of EV at the appropriate column in epoch estimate and update the 'epoch_estimate'
        # key back to disagg_output_object

        ev_epoch = np.c_[ts_1d, ev_1d]

        disagg_output_object = write_estimate(disagg_output_object, ev_epoch, read_col_idx, ev_out_idx,
                                              'epoch')

    # Stop the time counter for EV algo

    time_ev_end = datetime.now()

    # Log the runtime of EV module

    logger_ev.info('EV Estimation took | %0.3f s', get_time_diff(time_ev_start, time_ev_end))

    # Write exit status, code runtime, confidence level to the disagg_output_object
    # 'time'        : Runtime of module in seconds
    # 'confidence'  : Detection confidence values
    # 'exit_status' : Boolean to mark if any error encountered

    disagg_metrics_dict = {
        'time': get_time_diff(time_ev_start, time_ev_end),
        'confidence': 1.0,
        'exit_status': exit_status,
    }

    # Write the metrics to disagg output object

    disagg_output_object['disagg_metrics']['ev'] = disagg_metrics_dict

    # Remove the Keras models of L1 and L2 to avoid thread lock situations further in the pipeline

    disagg_input_object = remove_ev_cnn_models(detection_model_present, disagg_input_object)

    if not (disagg_mode == 'mtd'):

        disagg_output_object = \
            get_ev_profile(disagg_input_object, disagg_output_object, logger_ev_pass, debug)

    return disagg_input_object, disagg_output_object
