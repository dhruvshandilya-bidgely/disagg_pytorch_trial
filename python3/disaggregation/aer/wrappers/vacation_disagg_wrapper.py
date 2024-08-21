"""
Author - Mayank Sharan
Date - 28/11/18
Call the vacation disaggregation module and get results
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# Import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff

from python3.disaggregation.aer.vacation.vacation_disagg import vacation_disagg
from python3.disaggregation.aer.vacation.functions.wrapper_helpers import get_vacation_inputs
from python3.disaggregation.aer.vacation.functions.wrapper_helpers import write_vacation_results


def vacation_disagg_wrapper(disagg_input_object, disagg_output_object):

    """
    Wrapper code to provide interface to the pipeline to call vacation disaggregation code

    Parameters:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    Returns:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # ------------------------------------------- STAGE 1: INITIALISATIONS ---------------------------------------------

    # Initiate logger for the vacation module

    logger_vacation_base = disagg_input_object.get('logger').getChild('vacation_disagg_wrapper')
    logger_vacation = logging.LoggerAdapter(logger_vacation_base, disagg_input_object.get('logging_dict'))

    # Initialize logger pass dictionary to be passed along

    logger_pass = {
        'base_logger': logger_vacation_base,
        'logging_dict': disagg_input_object.get('logging_dict')
    }

    # Record the start time of processing

    t_vacation_start = datetime.now()

    # Initialize variables to avoid errors later

    debug = None
    type_1_epoch = np.empty(shape=(0, 2))
    type_2_epoch = np.empty(shape=(0, 2))

    # ------------------------------------------- STAGE 2: INPUT PREPARATION -------------------------------------------

    # Extract global config to decide the run

    global_config = disagg_input_object.get('config')

    # Get arguments to provide to the disagg code

    input_data, vacation_config, timed_disagg_output, exit_status = get_vacation_inputs(disagg_input_object,
                                                                                        disagg_output_object)

    # ------------------------------------------- STAGE 3: RUN MODULE --------------------------------------------------

    if global_config.get('run_mode') == 'prod' or global_config.get('run_mode') == 'custom':
        if global_config.get('disagg_mode') == 'historical':

            logger_vacation.info('Running vacation in historical mode |')
            debug, type_1_epoch, type_2_epoch = vacation_disagg(input_data, vacation_config, timed_disagg_output,
                                                                logger_pass)

        elif global_config.get('disagg_mode') == 'incremental':

            logger_vacation.info('Running vacation in incremental mode |')
            debug, type_1_epoch, type_2_epoch = vacation_disagg(input_data, vacation_config, timed_disagg_output,
                                                                logger_pass)

        elif global_config.get('disagg_mode') == 'mtd':

            logger_vacation.info('Running vacation in mtd mode |')
            debug, type_1_epoch, type_2_epoch = vacation_disagg(input_data, vacation_config, timed_disagg_output,
                                                                logger_pass)

        else:
            logger_vacation.error('Unrecognized disagg mode %s', global_config.get('disagg_mode'))

    # ------------------------------------------- STAGE 4: WRITE RESULTS -----------------------------------------------

    # Write results of vacation detection

    disagg_output_object = write_vacation_results(disagg_input_object, disagg_output_object, debug, type_1_epoch,
                                                  type_2_epoch, global_config, vacation_config, logger_vacation)

    t_vacation_end = datetime.now()

    logger_vacation.info('Vacation Estimation took | %.3f s ', get_time_diff(t_vacation_start, t_vacation_end))

    # Write exit status time taken etc.

    disagg_metrics_dict = {
        'time': get_time_diff(t_vacation_start, t_vacation_end),
        'confidence': 1.0,
        'exit_status': exit_status,
    }

    disagg_output_object['disagg_metrics']['va'] = disagg_metrics_dict

    return disagg_input_object, disagg_output_object
