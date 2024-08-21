"""
Author - Nisha Agarwal
Date - 8th Oct 20
Wrapper file for hybrid cooking
"""

# Import python packages

import logging
from datetime import datetime

# import functions from within the project

from python3.utils.write_estimate import write_estimate

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.cooking.cooking_module import run_cooking_module


def cooking_hybrid_wrapper(item_input_object, item_output_object, disagg_output_object):

    """
    Parameters:
        item_intput_object  (dict)              : Dictionary containing all hybrid inputs
        item_output_object  (dict)              : Dictionary containing all hybrid outputs
        disagg_output_object(dict)              : Dictionary containing all disagg outputs

    Returns:
        item_intput_object  (dict)              : Dictionary containing all hybrid inputs
        item_output_object  (dict)              : Dictionary containing all hybrid outputs
        disagg_output_object(dict)              : Dictionary containing all disagg outputs
    """

    # Initiate logger for the cooking module

    logger_cooking_base = item_input_object.get('logger').getChild('cooking_hybrid_wrapper')
    logger_cooking = logging.LoggerAdapter(logger_cooking_base, item_input_object.get('logging_dict'))
    logger_cooking_pass = {
        'logger_base': logger_cooking_base,
        'logging_dict': item_input_object.get('logging_dict'),
    }

    t_cooking_start = datetime.now()

    # Initialise arguments to give to the disagg code

    global_config = item_input_object.get('config')

    # Generate flags to determine on conditions if cooking should be run

    fail = False

    # Code flow can split here to accommodate separate run options for prod and custom

    if (global_config.get('run_mode') == 'prod' or global_config.get('run_mode') == 'custom'):
        if global_config.get('disagg_mode') in ['historical', 'incremental', 'mtd']:

            # Call the cooking hybrid module

            item_input_object, item_output_object, monthly_cooking, cooking_epoch = \
                run_cooking_module(item_input_object, item_output_object, logger_cooking_pass)

        else:
            fail = True
            logger_cooking.error('Unrecognized disagg mode %s |', global_config.get('disagg_mode'))
    else:
        fail = True
        logger_cooking.error('Unrecognized run mode %s |', global_config.get('disagg_mode'))

    if not fail:
        # Code to write results to disagg output object

        cooking_out_idx = item_output_object.get('output_write_idx_map').get('cook')
        cooking_read_idx = 1

        monthly_output_log = [(datetime.utcfromtimestamp(monthly_cooking[i, 0]).strftime('%b-%Y'), monthly_cooking[i, 1])
                              for i in range(monthly_cooking.shape[0])]

        logger_cooking.info('The monthly hybrid cooking consumption (in Wh) is : | %s', str(monthly_output_log).replace('\n', ' '))

        # Write billing cycle estimate

        item_output_object = write_estimate(item_output_object, monthly_cooking, cooking_read_idx, cooking_out_idx, 'bill_cycle')

        item_output_object = write_estimate(item_output_object, cooking_epoch, cooking_read_idx, cooking_out_idx, 'epoch')

    t_cooking_end = datetime.now()


    # Write exit status time taken etc.
    metrics = {
        'time': get_time_diff(t_cooking_start, t_cooking_end),
        'confidence': 1.0,
        'exit_status': 1,
    }

    item_output_object['disagg_metrics']['cook'] = metrics

    logger_cooking.info('Cooking Estimation took | %.3f s ', get_time_diff(t_cooking_start, t_cooking_end))

    return item_input_object, item_output_object, disagg_output_object
