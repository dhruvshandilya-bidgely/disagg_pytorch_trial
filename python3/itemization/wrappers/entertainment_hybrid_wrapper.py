"""
Author - Nisha Agarwal
Date - 8th Oct 20
Wrapper file for hybrid entertainment
"""

# Import python packages

import logging
from datetime import datetime

# import functions from within the project

from python3.utils.write_estimate import write_estimate

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.entertainment.entertainment_module import run_entertainment_module


def entertainment_hybrid_wrapper(item_input_object, item_output_object, disagg_output_object):

    """
    Parameters:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs

    Returns:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # Initiate logger for the entertainment module

    logger_entertainment_base = item_input_object.get('logger').getChild('entertainment_hybrid_wrapper')
    logger_entertainment = logging.LoggerAdapter(logger_entertainment_base, item_input_object.get('logging_dict'))
    logger_entertainment_pass = {
        'logger_base': logger_entertainment_base,
        'logging_dict': item_input_object.get('logging_dict'),
    }

    t_entertainment_start = datetime.now()

    # Initialise arguments to give to the disagg code

    global_config = item_input_object.get('config')

    # Generate flags to determine on conditions if entertainment should be run

    fail = False

    # Run entertainment disagg with different inputs as per requirement as for different modes

    # Code flow can split here to accommodate separate run options for prod and custom

    if (global_config.get('run_mode') == 'prod' or global_config.get('run_mode') == 'custom'):
        if global_config.get('disagg_mode') in ['historical', 'incremental', 'mtd']:

            # Call the entertainment hybrid module

            item_input_object, item_output_object, monthly_entertainment, entertainment_epoch = \
                run_entertainment_module(item_input_object, item_output_object, logger_entertainment_pass)

        else:
            fail = True
            logger_entertainment.error('Unrecognized disagg mode %s |', global_config.get('disagg_mode'))
    else:
        fail = True
        logger_entertainment.error('Unrecognized run mode %s |', global_config.get('disagg_mode'))

    if not fail:
        # Code to write results to disagg output object

        entertainment_out_idx = item_output_object.get('output_write_idx_map').get('ent')
        entertainment_read_idx = 1

        monthly_output_log = [(datetime.utcfromtimestamp(monthly_entertainment[i, 0]).strftime('%b-%Y'), monthly_entertainment[i, 1])
                              for i in range(monthly_entertainment.shape[0])]

        logger_entertainment.info('The monthly hybrid entertainment consumption (in Wh) is : | %s',
                                  str(monthly_output_log).replace('\n', ' '))

        # Write billing cycle estimate

        item_output_object = write_estimate(item_output_object, monthly_entertainment, entertainment_read_idx,
                                            entertainment_out_idx, 'bill_cycle')

        item_output_object = write_estimate(item_output_object, entertainment_epoch, entertainment_read_idx, entertainment_out_idx, 'epoch')

    t_entertainment_end = datetime.now()

    # Write exit status time taken etc.
    metrics = {
        'time': get_time_diff(t_entertainment_start, t_entertainment_end),
        'confidence': 1.0,
        'exit_status': 1,
    }

    item_output_object['disagg_metrics']['ent'] = metrics

    logger_entertainment.info('Lighting Estimation took | %.3f s ', get_time_diff(t_entertainment_start, t_entertainment_end))

    return item_input_object, item_output_object, disagg_output_object
