"""
Author - Nisha Agarwal
Date - 8th Oct 20
Wrapper file for hybrid lighting
"""

# Import python packages

import copy
import logging
import traceback
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.write_estimate import write_estimate

from python3.config.mappings.get_app_id import get_app_id

from python3.utils.time.get_time_diff import get_time_diff

from python3.utils.prepare_bc_tou_for_profile import prepare_bc_tou_for_profile

from python3.itemization.aer.lighting.lighting_module import run_lighting_module

from python3.utils.validate_output_schema import validate_appliance_profile_schema_for_billcycle


def extract_hsm(item_input_object, global_config):

    """
    Utility to extract hsm
        Parameters:
            item_input_object     (dict)              : Dictionary containing all inputs
            global_config         (dict)              : pipeline config
        Returns:
            hsm_in                (dict)              : extracted HSM
            hsm_fail              (bool)              : boolean to check whether extracted hsm is valid
    """

    # noinspection PyBroadException
    try:
        hsm_dic = item_input_object.get('appliances_hsm')
        hsm_in = hsm_dic.get('li')
    except KeyError:
        hsm_in = None

    hsm_fail = (hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0) and \
               (global_config.get("disagg_mode") == "mtd")

    return hsm_in, hsm_fail


def populate_lighting_user_profile(item_input_object, item_output_object, monthly_lighting, logger_pass):

    """
    Populate the lighting user profile object by bill cycle
    Parameters:
        item_input_object     (dict)              : Dictionary containing all inputs
        item_output_object    (dict)              : Dictionary containing all outputs
        debug                   (dict)              : Dictionary containing all intermediate debug variables
        logger_pass             (dict)              : Contains base logger and logging dictionary
    Returns:
        disagg_output_object    (dict)              : Dictionary containing all outputs
    """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('populate_lighting_user_profile')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    # Initialize constants to be used

    bc_start_col = 0
    bc_end_col = 1
    li_cons_col = 1

    li_app_id = get_app_id('li')

    # Initialize variables needed to populate bc level time of usage

    out_bill_cycles = item_input_object.get('out_bill_cycles')

    debug = item_output_object.get("debug").get("lighting_module_dict")

    # Prepare 1d data for TOU calculation

    lighting_ts_estimate = item_output_object.get("debug").get("lighting_module_dict").get("lighting_ts_estimate")

    input_data = copy.deepcopy(item_input_object.get('input_data'))
    lighting_input_aligned = np.zeros(shape=(input_data.shape[0], 2))

    lighting_input_aligned[:, 0] = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]

    _, idx_1, idx_2 = np.intersect1d(lighting_input_aligned[:, 0], lighting_ts_estimate[:, 0], return_indices=True)

    lighting_input_aligned[idx_1, 1] = lighting_ts_estimate[idx_2, 1]

    # dump_output(item_input_object, item_output_object)

    tou_dict = prepare_bc_tou_for_profile(input_data, lighting_input_aligned[:, 1], out_bill_cycles)

    # Extract variables from debug object needed to populate TOU

    for bill_cycle_idx in range(out_bill_cycles.shape[0]):

        # Extract the bill cycle to populate the profile for

        bill_cycle_start = out_bill_cycles[bill_cycle_idx, bc_start_col]
        bill_cycle_end = out_bill_cycles[bill_cycle_idx, bc_end_col]

        logger.debug('Lighting appliance profile population started for | %d', bill_cycle_start)

        try:

            # Initialize the dictionary containing the user profile attributes

            user_profile_object = dict(
                {
                    "validity": None,
                    "isPresent": True,
                    "detectionConfidence": 1.0,
                    "count": None,
                    "attributes": {
                        "lightingConsumption": None,
                        "morningCapacity": None,
                        "eveningCapacity": None,
                        "timeOfUsage": None,
                    },
                    "debugAttributes": {}
                }
            )

            # Fill morning and evening capacity values

            user_profile_object['attributes']['morningCapacity'] = float(np.round(debug.get("lighting_capacity"), 2))
            user_profile_object['attributes']['eveningCapacity'] = float(np.round(debug.get("lighting_capacity"), 2))

            user_profile_object['validity'] = dict()
            user_profile_object['validity']['start'] = int(bill_cycle_start)
            user_profile_object['validity']['end'] = int(bill_cycle_end)

            # Populate consumption for the bill cycle

            bc_row_bool = monthly_lighting[:, bc_start_col] == bill_cycle_start
            bc_lighting_cons = monthly_lighting[bc_row_bool, li_cons_col]

            if len(bc_lighting_cons) == 0:
                bc_lighting_cons = 0

            user_profile_object['attributes']['lightingConsumption'] = float(np.round(bc_lighting_cons, 2))

            user_profile_object['attributes']["timeOfUsage"] = list(tou_dict[bill_cycle_start])

            user_profile_list = [copy.deepcopy(user_profile_object)]

            # Populate appliance profile for the given bill cycle

            item_output_object['appliance_profile'][bill_cycle_start]['profileList'][0][str(li_app_id)] = \
                user_profile_list

        except Exception:

            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger.error('Lighting Profile Fill for billcycle %d failed | %s' % (int(bill_cycle_start), error_str))
            logger.info('Lighting Profile Empty for billcycle | %d ', int(bill_cycle_start))

        # Schema Validation for filled appliance profile

        validate_appliance_profile_schema_for_billcycle(item_output_object, bill_cycle_start, logger_pass)

        logger.debug('Lighting appliance profile population completed for | %d', bill_cycle_start)

    return item_output_object


def lighting_hybrid_wrapper(item_input_object, item_output_object, disagg_output_object):

    """
    Parameters:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs

    Returns:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # Initiate logger for the lighting module

    logger_lighting_base = item_input_object.get('logger').getChild('lighting_hybrid_wrapper')
    logger_lighting = logging.LoggerAdapter(logger_lighting_base, item_input_object.get('logging_dict'))
    logger_lighting_pass = {
        'logger_base': logger_lighting_base,
        'logging_dict': item_input_object.get('logging_dict'),
    }

    t_lighting_start = datetime.now()

    # Initialise arguments to give to the disagg code

    global_config = item_input_object.get('config')

    # Generate flags to determine on conditions if lighting should be run

    hsm_in, hsm_fail = extract_hsm(item_input_object, global_config)

    fail = False

    # Run lighting disagg with different inputs as per requirement as for different modes

    # Code flow can split here to accommodate separate run options for prod and custom

    if (global_config.get('run_mode') == 'prod' or global_config.get('run_mode') == 'custom'):
        if global_config.get('disagg_mode') == 'historical' or global_config.get('disagg_mode') == 'incremental':

            # Call the lighting hybrid module

            item_input_object, item_output_object, hsm, monthly_lighting, lighting_epoch = \
                run_lighting_module(item_input_object, item_output_object, global_config.get('disagg_mode'), hsm_in, logger_lighting_pass)

            if disagg_output_object.get('created_hsm') is not None and disagg_output_object.get('created_hsm').get('li') is not None:
                disagg_output_object['created_hsm']['li']['attributes'].update(hsm.get('attributes'))
                item_output_object['created_hsm']['li']['attributes'].update(hsm.get('attributes'))
            else:
                disagg_output_object['created_hsm']['li'] = hsm
                item_output_object['created_hsm']['li'] = hsm

            item_output_object = \
                populate_lighting_user_profile(item_input_object, item_output_object, monthly_lighting, logger_lighting_pass)

        elif global_config.get('disagg_mode') == 'mtd':

            # Call the lighting disagg module

            item_input_object, item_output_object, hsm, monthly_lighting, lighting_epoch = \
                run_lighting_module(item_input_object, item_output_object, global_config.get('disagg_mode'), hsm_in, logger_lighting_pass)

        else:
            fail = True
            logger_lighting.error('Unrecognized disagg mode %s |', global_config.get('disagg_mode'))
    else:
        fail = True
        logger_lighting.error('Unrecognized run mode %s |', global_config.get('disagg_mode'))

    if not fail:
        # Code to write results to disagg output object

        lighting_out_idx = item_output_object.get('output_write_idx_map').get('li')
        lighting_read_idx = 1

        monthly_output_log = [(datetime.utcfromtimestamp(monthly_lighting[i, 0]).strftime('%b-%Y'), monthly_lighting[i, 1])
                              for i in range(monthly_lighting.shape[0])]

        logger_lighting.info('The monthly hybrid lighting consumption (in Wh) is : | %s', str(monthly_output_log).replace('\n', ' '))

        # Write billing cycle estimate

        item_output_object = write_estimate(item_output_object, monthly_lighting, lighting_read_idx, lighting_out_idx, 'bill_cycle')

        item_output_object = write_estimate(item_output_object, lighting_epoch, lighting_read_idx, lighting_out_idx, 'epoch')

    t_lighting_end = datetime.now()

    # Write exit status time taken etc.
    metrics = {
        'time': get_time_diff(t_lighting_start, t_lighting_end),
        'confidence': 1.0,
        'exit_status': 1,
    }

    item_output_object['disagg_metrics']['li'] = metrics

    logger_lighting.info('Lighting Estimation took | %.3f s ', get_time_diff(t_lighting_start, t_lighting_end))

    return item_input_object, item_output_object, disagg_output_object
