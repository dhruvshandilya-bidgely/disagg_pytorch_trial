"""
Author - Nisha Agarwal
Date - 8th Oct 20
Wrapper file for hybrid laundry
"""

# Import python packages

import logging
from datetime import datetime

# import functions from within the project

from python3.utils.write_estimate import write_estimate

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.laundry.laundry_module import run_laundry_module


def extract_hsm(item_input_object, global_config):

    """
    Utility to extract hsm

    Parameters:
        item_input_object     (dict)              : Dictionary containing all inputs
        global_config         (dict)              : config dict

    Returns:
        hsm_in                (dict)              : hsm input
        hsm_fail              (bool)              : true if valid hsm is present
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


def laundry_hybrid_wrapper(item_input_object, item_output_object, disagg_output_object):

    """
    wrapper for initializing laundry estimates

    Parameters:
        item_input_object     (dict)              : Dictionary containing all hybrid inputs
        item_output_object    (dict)              : Dictionary containing all hybrid outputs
        disagg_output_object  (dict)              : Dictionary containing all disagg outputs

    Returns:
        item_input_object     (dict)              : Dictionary containing all hybrid inputs
        item_output_object    (dict)              : Dictionary containing all hybrid outputs
        disagg_output_object  (dict)              : Dictionary containing all disagg outputs
    """

    # Initiate logger for the laundry module

    logger_laundry_base = item_input_object.get('logger').getChild('laundry_hybrid_wrapper')
    logger_laundry = logging.LoggerAdapter(logger_laundry_base, item_input_object.get('logging_dict'))
    logger_laundry_pass = {
        'logger_base': logger_laundry_base,
        'logging_dict': item_input_object.get('logging_dict'),
    }

    t_laundry_start = datetime.now()

    # Initialise arguments to give to the disagg code

    global_config = item_input_object.get('config')

    # Generate flags to determine on conditions if laundry should be run

    hsm_in, hsm_fail = extract_hsm(item_input_object, global_config)

    fail = False

    # Run laundry disagg with different inputs as per requirement as for different modes

    # Code flow can split here to accommodate separate run options for prod and custom

    if global_config.get('run_mode') in ['custom', 'prod']:
        if global_config.get('disagg_mode') in ['historical', 'incremental']:

            # Call the laundry hybrid module

            item_input_object, item_output_object, hsm, monthly_laundry, laundry_epoch = \
                run_laundry_module(item_input_object, item_output_object, global_config.get('disagg_mode'), hsm_in, logger_laundry_pass)

            if disagg_output_object.get('created_hsm') is not None and disagg_output_object.get('created_hsm').get('li') is not None:

                if hsm.get('attributes') is not None and hsm.get('attributes').get('attributes') is not None:
                    disagg_output_object['created_hsm']['li']['attributes'].update(hsm.get('attributes').get('attributes'))
                    item_output_object['created_hsm']['li']['attributes'].update(hsm.get('attributes').get('attributes'))
                else:
                    disagg_output_object['created_hsm']['li']['attributes'].update(hsm.get('attributes'))
                    item_output_object['created_hsm']['li']['attributes'].update(hsm.get('attributes'))
            else:
                disagg_output_object['created_hsm']['li'] = hsm
                item_output_object['created_hsm']['li'] = hsm

        elif global_config.get('disagg_mode') == 'mtd':

            # Call the laundry disagg module

            item_input_object, item_output_object, hsm, monthly_laundry, laundry_epoch = \
                run_laundry_module(item_input_object, item_output_object, global_config.get('disagg_mode'), hsm_in, logger_laundry_pass)

        else:
            fail = True
            logger_laundry.error('Unrecognized disagg mode %s |', global_config.get('disagg_mode'))
    else:
        fail = True
        logger_laundry.error('Unrecognized run mode %s |', global_config.get('disagg_mode'))

    if not fail:
        # Code to write results to disagg output object

        laundry_out_idx = item_output_object.get('output_write_idx_map').get('ld')
        laundry_read_idx = 1

        monthly_output_log = [(datetime.utcfromtimestamp(monthly_laundry[i, 0]).strftime('%b-%Y'), monthly_laundry[i, 1])
                              for i in range(monthly_laundry.shape[0])]

        logger_laundry.info('The monthly hybrid laundry consumption (in Wh) is : | %s',
                            str(monthly_output_log).replace('\n', ' '))

        # Write billing cycle estimate

        item_output_object = write_estimate(item_output_object, monthly_laundry, laundry_read_idx, laundry_out_idx, 'bill_cycle')

        item_output_object = write_estimate(item_output_object, laundry_epoch, laundry_read_idx, laundry_out_idx, 'epoch')

    t_laundry_end = datetime.now()

    # Write exit status time taken etc.
    metrics = {
        'time': get_time_diff(t_laundry_start, t_laundry_end),
        'confidence': 1.0,
        'exit_status': 1,
    }

    item_output_object['disagg_metrics']['ld'] = metrics

    logger_laundry.info('Lighting Estimation took | %.3f s ', get_time_diff(t_laundry_start, t_laundry_end))

    return item_input_object, item_output_object, disagg_output_object
