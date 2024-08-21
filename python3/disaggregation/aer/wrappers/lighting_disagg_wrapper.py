"""
Author - Mayank Sharan
Date - 01/10/18
Call the lighting disaggregation module and get results
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.write_estimate import write_estimate
from python3.config.pilot_constants import PilotConstants
from python3.utils.time.get_time_diff import get_time_diff

from python3.disaggregation.aer.lighting.lighting_disagg import lighting_disagg
from python3.disaggregation.aer.lighting.lighting_disagg_jp import lighting_disagg_jp
from python3.disaggregation.aer.lighting.init_lighting_params import init_lighting_params
from python3.disaggregation.aer.lighting.functions.populate_lighting_user_profile import populate_lighting_user_profile


def extract_vacation(disagg_output_object):

    """Utility function to extract vacation_periods"""

    # Extract vacation period from disagg output object

    vacation_periods = disagg_output_object.get('special_outputs').get('vacation_periods')

    if vacation_periods is None:
        vacation_periods = np.array([])

    return vacation_periods


def extract_hsm(disagg_input_object, global_config):
    """Utility to extract hsm"""

    # noinspection PyBroadException
    try:
        hsm_dic = disagg_input_object.get('appliances_hsm')
        hsm_in = hsm_dic.get('li')
    except KeyError:
        hsm_in = None

    hsm_fail = (hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0) and \
               (global_config.get("disagg_mode") == "mtd")

    return hsm_in, hsm_fail


def call_lighting_module(disagg_input_object, input_data, lighting_config, vacation_periods, hist_mode, hsm_in,
                         bypass_hsm, logger_pass, logger_lighting):

    """Utility function to call lighting module depending upon the pilot id"""

    # Extract Pilot Id to decide which lighting module to run

    pilot_id = disagg_input_object.get('config').get('pilot_id')

    # Based on pilot id decide lighting run

    if pilot_id in PilotConstants.HVAC_JAPAN_PILOTS:

        # for Japan pilot ID

        logger_lighting.info('Running Japan lighting. Pilot id | %d ', pilot_id)

        monthly_lighting, ts, debug, debug_hsm, hsm, exit_status = lighting_disagg_jp(input_data, lighting_config,
                                                                                      vacation_periods, hist_mode,
                                                                                      hsm_in, bypass_hsm, logger_pass)
    else:

        logger_lighting.info('Running standard lighting. Pilot id | %d ', pilot_id)

        # Call normal lighting module here

        monthly_lighting, ts, debug, debug_hsm, hsm, exit_status = lighting_disagg(input_data, lighting_config,
                                                                                   vacation_periods, hist_mode,
                                                                                   hsm_in, bypass_hsm, logger_pass)

    return monthly_lighting, ts, debug, debug_hsm, hsm, exit_status


def lighting_disagg_wrapper(disagg_input_object, disagg_output_object):

    """
    Parameters:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs

    Returns:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # Initiate logger for the lighting module

    logger_lighting_base = disagg_input_object.get('logger').getChild('lighting_disagg_wrapper')
    logger_lighting = logging.LoggerAdapter(logger_lighting_base, disagg_input_object.get('logging_dict'))
    logger_pass = {
        'logger_base': logger_lighting_base,
        'logging_dict': disagg_input_object.get('logging_dict'),
    }

    t_lighting_start = datetime.now()

    # Initialise arguments to give to the disagg code

    global_config = disagg_input_object.get('config')
    input_data = copy.deepcopy(disagg_input_object.get('input_data'))

    lighting_config = init_lighting_params()
    lighting_config['UUID'] = global_config.get('uuid')

    vacation_periods = extract_vacation(disagg_output_object)

    # Initialize outputs so that we don't get errors

    monthly_lighting = np.array([])
    lighting_epoch = np.array([])

    exit_status = {
        'exit_code': 1,
        'error_list': [],
    }

    # Generate flags to determine on conditions if lighting should be run

    hsm_in, hsm_fail = extract_hsm(disagg_input_object, global_config)

    days_of_data = (input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX] -
                    input_data[0, Cgbdisagg.INPUT_EPOCH_IDX]) / Cgbdisagg.SEC_IN_DAY

    has_min_days = (days_of_data > lighting_config.get('DURATION_VALID_DISAGG')) or \
                   (global_config.get('disagg_mode') not in ['historical', 'incremental'])

    # Run lighting disagg with different inputs as per requirement as for different modes

    # Code flow can split here to accommodate separate run options for prod and custom

    if (global_config.get('run_mode') == 'prod' or global_config.get('run_mode') == 'custom') and not hsm_fail:
        if global_config.get('disagg_mode') == 'historical' and has_min_days:

            # Initialise parameters as per the mode

            hist_mode = True

            # This is a major bug, Bypass HSM should be true for historical mode

            bypass_hsm = False

            # Call the lighting disagg module

            monthly_lighting, ts, debug, debug_hsm, hsm, exit_status = \
                call_lighting_module(disagg_input_object, input_data, lighting_config, vacation_periods, hist_mode,
                                     hsm_in, bypass_hsm, logger_pass, logger_lighting)

            # Set the hsm and get epoch level estimate

            disagg_output_object['created_hsm']['li'] = hsm

            ts_1d = np.reshape(ts, newshape=(len(ts) * len(ts[0]),))

            lighting_1d = debug.get('data').get('lighting')
            lighting_1d = np.reshape(lighting_1d, newshape=(len(lighting_1d) * len(lighting_1d[0]),))

            lighting_epoch = np.c_[ts_1d, lighting_1d]

            disagg_output_object = populate_lighting_user_profile(disagg_input_object, disagg_output_object, debug,
                                                                  logger_pass)

        elif global_config.get('disagg_mode') == 'incremental' and has_min_days:

            # Initialise parameters as per the mode

            hist_mode = True
            bypass_hsm = False

            # Call the lighting disagg module

            monthly_lighting, ts, debug, debug_hsm, hsm, exit_status = \
                call_lighting_module(disagg_input_object, input_data, lighting_config, vacation_periods, hist_mode,
                                     hsm_in, bypass_hsm, logger_pass, logger_lighting)

            # Set the hsm and get epoch level estimate

            disagg_output_object['created_hsm']['li'] = hsm

            ts_1d = np.reshape(ts, newshape=(len(ts) * len(ts[0]),))

            lighting_1d = debug_hsm.get('data').get('lighting')
            lighting_1d = np.reshape(lighting_1d, newshape=(len(lighting_1d) * len(lighting_1d[0]),))

            lighting_epoch = np.c_[ts_1d, lighting_1d]

            disagg_output_object = populate_lighting_user_profile(disagg_input_object, disagg_output_object, debug,
                                                                  logger_pass)

        elif global_config.get('disagg_mode') == 'mtd':

            # Initialise parameters as per the mode

            hist_mode = False
            bypass_hsm = False

            # Call the lighting disagg module

            monthly_lighting, ts, debug, debug_hsm, hsm, exit_status = \
                call_lighting_module(disagg_input_object, input_data, lighting_config, vacation_periods, hist_mode,
                                     hsm_in, bypass_hsm, logger_pass, logger_lighting)

            # Get the epoch level estimate

            ts_1d = np.reshape(ts, newshape=(len(ts) * len(ts[0]),))

            lighting_1d = debug_hsm.get('data').get('lighting')
            lighting_1d = np.reshape(lighting_1d, newshape=(len(lighting_1d) * len(lighting_1d[0]),))

            lighting_epoch = np.c_[ts_1d, lighting_1d]

        elif not has_min_days:
            logger_lighting.info('Lighting disagg not done since number of days are less than %d |',
                                 lighting_config.get('DURATION_VALID_DISAGG'))
        else:
            logger_lighting.error('Unrecognized disagg mode %s |', global_config.get('disagg_mode'))

    if not hsm_fail and has_min_days:

        # Code to write results to disagg output object

        lighting_out_idx = disagg_output_object.get('output_write_idx_map').get('li')
        lighting_read_idx = 1

        # Writing the monthly output to log

        monthly_output_log = [(datetime.utcfromtimestamp(monthly_lighting[i, 0]).strftime('%b-%Y'),
                               monthly_lighting[i, 1]) for i in range(monthly_lighting.shape[0])]
        logger_lighting.info("The monthly lighting consumption (in Wh) is : | %s",
                             str(monthly_output_log).replace('\n', ' '))

        # Write billing cycle estimate

        disagg_output_object = write_estimate(disagg_output_object, monthly_lighting, lighting_read_idx,
                                              lighting_out_idx, 'bill_cycle')

        # Write timestamp level estimates

        disagg_output_object = write_estimate(disagg_output_object, lighting_epoch, lighting_read_idx, lighting_out_idx,
                                              'epoch')

    if hsm_fail:
        logger_lighting.warning('Lighting did not run since %s mode required HSM and HSM was missing |',
                                global_config.get('disagg_mode'))

    t_lighting_end = datetime.now()

    logger_lighting.info('Lighting Estimation took | %.3f s ', get_time_diff(t_lighting_start, t_lighting_end))

    # Write exit status time taken etc.

    disagg_metrics_dict = {
        'time': get_time_diff(t_lighting_start, t_lighting_end),
        'confidence': 1.0,
        'exit_status': exit_status,
    }

    disagg_output_object['disagg_metrics']['li'] = disagg_metrics_dict

    return disagg_input_object, disagg_output_object
