"""
Author - Nikhil Singh Chauhan
Date - 16th Oct 2018
Call the WaterHeater disaggregation module and write output
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.write_estimate import write_estimate
from python3.utils.time.get_time_diff import get_time_diff

from python3.disaggregation.aer.waterheater.water_heater_disagg import water_heater_disagg

from python3.initialisation.load_files.load_wh_models import get_waterheater_models

from python3.disaggregation.aer.waterheater.functions.output_validation import check_epoch_output
from python3.disaggregation.aer.waterheater.functions.output_validation import check_month_output

from python3.disaggregation.aer.waterheater.functions.initialize_debug import initialize_debug
from python3.disaggregation.aer.waterheater.functions.get_wh_app_profile import get_wh_app_profile
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.hsm_utils import get_hsm

from python3.disaggregation.aer.waterheater.functions.water_heater_utils import get_other_appliance_output

from python3.disaggregation.aer.waterheater.functions.get_waterheater_user_profile import get_waterheater_profile

from python3.disaggregation.aer.waterheater.functions.initialize_waterheater_params import initialize_water_heater_params


def waterheater_disagg_wrapper(disagg_input_object, disagg_output_object):
    """
    Parameters:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs

    Returns:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # Initiate logger for the water heater module

    logger_water_heater_base = disagg_input_object.get('logger').getChild('wh_disagg_wrapper')
    logger_water_heater_pass = {'logger': logger_water_heater_base,
                                'logging_dict': disagg_input_object.get('logging_dict')}

    # Function specific logger

    logger_water_heater = logging.LoggerAdapter(logger_water_heater_base, disagg_input_object.get('logging_dict'))

    # Variable to store the status of algo run success/failure and the known errors
    # 'exit_code' : status of code run (1: Success, 0: Failure with known errors, -1: Failure with unknown errors)
    # 'error_list': List of handles errors encountered

    exit_status = {
        'exit_code': 0,
        'error_list': []
    }

    # Starting the algorithm time counter

    time_water_heater_start = datetime.now()

    # List to store the errors throughout the algorithm run

    error_list = []

    # Reading global configuration from disagg_input_object

    global_config = disagg_input_object.get('config')

    # Initialize the parameters required for Water Heater algorithm

    wh_config = initialize_water_heater_params(disagg_input_object, global_config)

    # Bill cycles for which output is to written (first column)

    out_bill_cycles = disagg_input_object.get('out_bill_cycles')[:, 0]

    # Get HSM (if present) and check for invalid hsm in this mode

    hsm_in, hsm_fail = get_hsm(disagg_input_object, global_config, logger_water_heater_pass)

    # Read the water heater models from input object

    models = get_waterheater_models(disagg_input_object, logger_water_heater_pass)

    # Initialize debug object

    debug = initialize_debug(global_config, hsm_in, out_bill_cycles, models)

    # Reading input data from disagg_input_object

    input_data = deepcopy(disagg_input_object.get('input_data'))

    # Replacing the NaNs that were made zero while pre-processing

    is_nan_cons = disagg_input_object['data_quality_metrics']['is_nan_cons']
    input_data[is_nan_cons, Cgbdisagg.INPUT_CONSUMPTION_IDX] = np.nan

    # Reading appliance profile of the user for water heater present

    wh_present = get_wh_app_profile(disagg_input_object, wh_config, logger_water_heater)

    # Putting together other appliance output and ground truth if needed in water heater

    debug = get_other_appliance_output(disagg_output_object, debug)

    # Store home meta data in debug object
    debug['home_meta_data'] = disagg_input_object.get('home_meta_data')

    # Get disagg_mode and run_mode from the input object

    run_mode = global_config.get('run_mode')
    disagg_mode = global_config.get('disagg_mode')

    # Check if water heater module to be run on this mode based on hsm and config

    if (run_mode == 'prod' or run_mode == 'custom') and (hsm_fail is False):
        # If run mode valid

        if (disagg_mode == 'historical') or (disagg_mode == 'incremental'):
            # Historical / Incremental mode

            # Calling the main water heater disagg module

            monthly_output, debug, hsm, exit_status = water_heater_disagg(input_data, wh_config, global_config,
                                                                          wh_present, debug, exit_status, error_list,
                                                                          logger_water_heater_pass)

            # Saving the new HSM to the disagg_output_object

            disagg_output_object['created_hsm']['wh'] = hsm

        elif disagg_mode == 'mtd':
            # MTD run mode

            # Calling the main water heater disagg module

            monthly_output, debug, hsm, exit_status = water_heater_disagg(input_data, wh_config, global_config,
                                                                          wh_present, debug, exit_status, error_list,
                                                                          logger_water_heater_pass)
        else:
            # If invalid run mode, define the default empty output

            debug = {}
            monthly_output = np.array([])

            logger_water_heater.error('Unrecognized disagg mode {} |'.format(disagg_mode))
    else:
        # If run mode valid but HSM not found for this mode

        debug = {}
        monthly_output = np.array([])

        logger_water_heater.info('Water heater algorithm did not run | ')

        logger_water_heater.info('Run mode | {}'.format(run_mode))
        logger_water_heater.info('HSM fail status | {}'.format(hsm_fail))

    # If valid HSM is there, write epoch/bill_cycle output to the disagg_output_object

    if not hsm_fail:

        # Read index from disagg output object for which output is to written

        water_heater_out_idx = disagg_output_object.get('output_write_idx_map').get('wh')
        read_col_idx = 1

        # Read the table containing billing cycle estimate for each appliance

        bill_cycle_est = disagg_output_object.get('bill_cycle_estimate')

        # Check if monthly estimate received from algo, else create dummy data

        monthly_output = check_month_output(monthly_output, bill_cycle_est, logger_water_heater)

        # Log the type of water heater

        if debug.get('timed_hld') == 1:
            wh_type = 'timed'
        elif debug.get('thermostat_hld') == 1:
            wh_type = 'non-timed'
        else:
            wh_type = 'none'

        logger_water_heater.info('Water heater type | {}'.format(wh_type))

        # Log the monthly output

        monthly_output_log = [(datetime.utcfromtimestamp(monthly_output[i, 0]).strftime('%b-%Y'), monthly_output[i, 1])
                              for i in range(monthly_output.shape[0])]

        logger_water_heater.info('The monthly WH consumption (in Wh) is : | %s',
                                 str(monthly_output_log).replace('\n', ' '))

        # Write the water heater estimate at the appropriate column and update the 'bill_cycle_estimate' in the
        # disagg output object

        disagg_output_object = write_estimate(disagg_output_object, monthly_output, read_col_idx, water_heater_out_idx,
                                              'bill_cycle')

        # Pick timestamps for which output is to written

        epoch_est = disagg_output_object.get('epoch_estimate')

        # Check if the epoch level received from the algo, else create dummy data

        ts_1d, wh_1d = check_epoch_output(debug, epoch_est, logger_water_heater)

        # Write the estimate of Water heaetr at the appropriate column in epoch estimate and update the 'epoch_estimate'
        # key back to disagg_output_object

        wh_epoch = np.c_[ts_1d, wh_1d]

        disagg_output_object = write_estimate(disagg_output_object, wh_epoch, read_col_idx, water_heater_out_idx,
                                              'epoch')

    else:
        # If water heater module did not run, log the values

        logger_water_heater.warning('Water Heater did not run because %s mode needs HSM and it is missing | ',
                                    disagg_mode)

    # Adding timed WH output to special outputs, for use in hybrid module

    disagg_output_object['special_outputs']['timed_water_heater'] = debug.get('timed_wh_signal')

    disagg_output_object = update_special_outputs(disagg_output_object, debug)

    if debug.get('timed_debug') is None:
        disagg_output_object['special_outputs']['timed_wh_confidence'] = 0.0
    else:
        disagg_output_object['special_outputs']['timed_wh_confidence'] = debug.get('timed_debug').get('timed_confidence')
        disagg_output_object['special_outputs']['timed_debug'] = debug.get('timed_debug')

    # Stop the time counter for Water Heater algo

    time_water_heater_end = datetime.now()

    # Log the runtime of water heater module

    logger_water_heater.info('Water Heater Estimation took | %0.3f s', get_time_diff(time_water_heater_start,
                                                                                     time_water_heater_end))

    # Add user attributes to the user profile object
    if not (disagg_mode == 'mtd'):
        disagg_output_object = get_waterheater_profile(disagg_input_object, disagg_output_object,
                                                       logger_water_heater_pass,
                                                       debug)

    # Write exit status, code runtime, confidence level to the disagg_output_object
    # 'time'        : Runtime of module in seconds
    # 'confidence'  : Detection confidence values
    # 'exit_status' : Boolean to mark if any error encountered

    disagg_metrics_dict = {
        'time': get_time_diff(time_water_heater_start, time_water_heater_end),
        'confidence': 1.0,
        'exit_status': exit_status,
    }

    # Write the metrics to disagg output object

    disagg_output_object['disagg_metrics']['wh'] = disagg_metrics_dict

    return disagg_input_object, disagg_output_object


def update_special_outputs(disagg_output_object, debug):

    """
    update special outputs with certain debug values, which will be used in hybrid
    Parameters:
        disagg_output_object(dict)              : Dictionary containing all outputs
        debug (dict)                            : wh debug dict

    Returns:
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    water_heater_out_idx = disagg_output_object.get('output_write_idx_map').get('wh')

    if disagg_output_object['special_outputs']['timed_water_heater'] is not None:
        disagg_output_object["epoch_estimate"][:,  water_heater_out_idx] = np.nan_to_num(disagg_output_object["epoch_estimate"][:,  water_heater_out_idx]) + \
                                                                           disagg_output_object['special_outputs']['timed_water_heater'][:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Adding thin pulse and fat pulse to special outputs

    if debug is not None:
        if debug.get('final_thin_output') is not None:
            disagg_output_object['special_outputs']['final_thin_pulse'] = np.nan_to_num(debug['final_thin_output'])

        if debug.get('final_fat_output') is not None:
            disagg_output_object['special_outputs']['final_fat_pulse'] = np.nan_to_num(debug['final_fat_output'])

    return disagg_output_object
