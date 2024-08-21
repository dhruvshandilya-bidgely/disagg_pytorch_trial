"""
Author - Sahana M
Date - 9/3/2021
This file contains the main function that calls the Water Heater algorithm and get the monthly, epoch level estimate
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy
from datetime import datetime

# import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.write_estimate import write_estimate
from python3.utils.time.get_time_diff import get_time_diff
from python3.itemization.aer.water_heater.seasonal_wh_module import seasonal_wh_module
from python3.itemization.aer.water_heater.functions.swh_appliance_profile import get_waterheater_profile
from python3.itemization.aer.water_heater.functions.create_outputs import create_ts_output, create_monthly_output
from python3.itemization.aer.water_heater.functions.output_validation import check_month_output, check_epoch_output, get_bill_cycle_info


def seasonal_wh_disagg(item_input_object, item_output_object, hsm_in, hsm_fail, logger_pass):

    """
    Wrapper file for seasonal wh
    Args:
        item_input_object               (dict)      : Dictionary containing all item inputs
        item_output_object              (dict)      : Dictionary containing all item outputs
        hsm_in                          (dict)      : HSM data
        hsm_fail                        (Boolean)   : HSM retrieval status
        logger_pass                     (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        itemisation_input_object         (dict)      : Dictionary containing all inputs
        itemisation_output_object        (dict)      : Dictionary containing all outputs
    """

    logger_base = logger_pass.get('logger').getChild('seasonal_wh_disagg')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    logger_pass['logger'] = logger_base

    # Initialise arguments to be given to the seasonal wh module

    global_config = item_input_object.get('config')
    global_config['home_meta_data'] = item_input_object.get('home_meta_data')

    # Initialise input data for seasonal wh module

    input_data = deepcopy(item_input_object.get('input_data'))

    # Variable to store the status of algo run success/failure and the known errors
    # 'exit_code' : status of code run (1: Success, 0: Failure with known errors, -1: Failure with unknown errors)
    # 'error_list': List of handles errors encountered

    exit_status_dict = {
        'exit_code': 1,
        'error_list': []
    }

    # Start the timer

    t_start = datetime.now()

    # Store vacation output in debug object

    debug = dict()

    # Store the hsm information

    debug['hsm_fail'] = hsm_fail
    debug['hsm_in'] = hsm_in
    debug['disagg_mode'] = item_input_object.get('config').get('disagg_mode')

    # For historical and incremental mode store the hsm

    debug['make_hsm'] = False
    if debug.get('disagg_mode') != 'mtd':
        debug['make_hsm'] = True

    vacation_indexes = deepcopy(item_input_object.get('disagg_output_write_idx_map').get('va'))
    vacation_indexes.append(0)
    debug['vacation_output'] = deepcopy(item_input_object.get('disagg_epoch_estimate')[:, vacation_indexes])

    ts_list = item_input_object.get('item_input_params').get('ts_list')

    # Get the out bill cycles & month ts in the debug object

    debug['out_bill_cycles'] = item_input_object['out_bill_cycles']
    debug['month_ts'] = item_input_object.get('item_input_params').get('month_ts')

    # Call the water heater module

    debug, exit_status = seasonal_wh_module(input_data, debug, global_config, logger_pass)

    # Only is the run was successful calculate time stamp and monthly output

    if not exit_status:

        # Update timestamp in hsm_in
        hsm_in = debug.get('hsm_in')

        hsm_in.update({
            'timestamp': ts_list[-1]
        })

        # Create Time stamp level output

        swh_ts_estimate, final_estimate = create_ts_output(input_data, debug)
        debug['final_wh_signal'] = final_estimate

        # Create Bill cycle level output

        final_monthly_estimate, monthly_output_log = create_monthly_output(final_estimate)

        logger.info('The monthly SWH consumption (in Wh) is : | %s', str(monthly_output_log).replace('\n', ' '))

        # Initialise item output objects

        item_output_object['special_outputs']['seasonal_wh'] = {
            'ts_estimate': swh_ts_estimate,
            'wh_potential': debug.get('wh_potential'),
            'final_e_range': debug.get('final_e_range'),
            'user_detection': debug.get('swh_hld'),
            'total_consumption': debug.get('final_consumption'),
            'total_tb_detections': debug.get('total_detections'),
        }

        for i in range(debug.get('total_detections')):
            item_output_object['special_outputs']['seasonal_wh'].update({
                str('tb_' + str(i) + '_score'): debug.get('swh_run' + str(i) + '_score'),
                str('tb_' + str(i) + '_start_time'): debug.get('swh_run' + str(i) + '_start_time'),
                str('tb_' + str(i) + '_end_time'): debug.get('swh_run' + str(i) + '_end_time')
            })

        debug['hsm_in'] = hsm_in

        # Update the monthly consumption in item_output_object

        item_output_object['special_outputs']['seasonal_wh'].update({
            'bill_cycle_estimate': final_monthly_estimate,
            'monthly_output': monthly_output_log
        })

        # --------------------------------------------- WRITE OUTPUTS --------------------------------------------------

        # Get monthly output

        debug = get_bill_cycle_info(input_data, debug)

        bill_cycle_ts = debug['bill_cycle_ts']
        bill_cycle_idx = debug['bill_cycle_idx']

        bill_cycle_consumption = np.bincount(bill_cycle_idx, final_estimate[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        monthly_output = np.c_[bill_cycle_ts, bill_cycle_consumption]

        # Read index from disagg output object for which output is to written

        water_heater_out_idx = item_output_object.get('output_write_idx_map').get('wh')
        read_col_idx = 1

        # Read the table containing billing cycle estimate for each appliance

        bill_cycle_est = item_output_object.get('bill_cycle_estimate')

        # Check if monthly estimate received from algo, else create dummy data

        monthly_output = check_month_output(monthly_output, bill_cycle_est, logger)

        # Log the type of water heater

        if debug.get('swh_hld') == 1:
            wh_type = 'seasonal-water-heater'
        else:
            wh_type = 'none'

        logger.info('Water heater type | {}'.format(wh_type))

        # Write the water heater estimate at the appropriate column and update the 'bill_cycle_estimate' in the
        # itemization output object and disagg output object

        item_output_object = write_estimate(item_output_object, monthly_output, read_col_idx, water_heater_out_idx,
                                            'bill_cycle')

        # Pick timestamps for which output is to written

        epoch_est = item_output_object.get('epoch_estimate')

        # Check if the epoch level received from the algo, else create dummy data

        ts_1d, wh_1d = check_epoch_output(debug, epoch_est, logger)

        # Write the estimate of Water heaetr at the appropriate column in epoch estimate

        wh_epoch = np.c_[ts_1d, wh_1d]

        item_output_object = write_estimate(item_output_object, wh_epoch, read_col_idx, water_heater_out_idx, 'epoch')

        # Add user attributes to the user profile object

        if not (debug.get('disagg_mode') == 'mtd'):
            item_output_object = get_waterheater_profile(item_input_object, item_output_object, logger_pass, debug)

        t_end = datetime.now()

        # Write exit status, code runtime, confidence level to the disagg_output_object
        # 'time'        : Runtime of module in seconds
        # 'confidence'  : Detection confidence values
        # 'exit_status' : Boolean to mark if any error encountered

        disagg_metrics_dict = {
            'time': get_time_diff(t_start, t_end),
            'confidence': 1.0,
            'exit_status': exit_status_dict,
        }

    else:

        t_end = datetime.now()

        exit_status_dict = {
            'exit_code': 0,
            'error_list': ['SWH failed due to some reasons']
        }

        # Write exit status, code runtime, confidence level to the disagg_output_object
        # 'time'        : Runtime of module in seconds
        # 'confidence'  : Detection confidence values
        # 'exit_status' : Boolean to mark if any error encountered

        disagg_metrics_dict = {
            'time': get_time_diff(t_start, t_end),
            'confidence': 1.0,
            'exit_status': exit_status_dict,
        }

    # Write the metrics to disagg output object

    item_output_object['itemization_metrics']['wh'] = disagg_metrics_dict

    return item_input_object, item_output_object, exit_status, debug
