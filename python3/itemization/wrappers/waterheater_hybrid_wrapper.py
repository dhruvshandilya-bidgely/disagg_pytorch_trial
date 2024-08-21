"""
Author - Sahana M
Date - 4/3/2021
Wrapper file for seasonal wh
"""

# Import python packages

import logging
from datetime import datetime

# import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff
from python3.itemization.aer.water_heater.functions.swh_hsm_utils import update_hsm
from python3.itemization.aer.water_heater.functions.swh_hsm_utils import extract_hsm
from python3.itemization.aer.water_heater.seasonal_wh_disagg import seasonal_wh_disagg
from python3.itemization.aer.water_heater.functions.get_detection_status import get_detection_status


def waterheater_hybrid_wrapper(item_input_object, item_output_object, pipeline_output_object):

    """
    Wrapper file for seasonal wh
    Args:
        item_input_object             (dict)      : Dictionary containing all hybrid inputs
        item_output_object            (dict)      : Dictionary containing all hybrid outputs
        pipeline_output_object            (dict)      : Dictionary containing all disagg outputs

    Returns:
        itemisation_input_object         (dict)      : Dictionary containing all inputs
        itemisation_output_object        (dict)      : Dictionary containing all outputs
    """

    # Initiate logger for the seasonal wh module

    logger_seasonal_wh_base = item_input_object.get('logger').getChild('waterheater_hybrid_wrapper')
    logger_seasonal_wh = logging.LoggerAdapter(logger_seasonal_wh_base, item_input_object.get('logging_dict'))
    logger_seasonal_wh_pass = {
        'logger': logger_seasonal_wh_base,
        'logging_dict': item_input_object.get('logging_dict'),
    }

    t_seasonal_wh_start = datetime.now()

    # Initialise arguments to be given to the seasonal wh module

    global_config = item_input_object.get('config')

    # Extract hsm

    hsm_in, hsm_fail = extract_hsm(item_input_object, global_config)

    # Get disagg_mode and run_mode from the input object

    run_mode = global_config.get('run_mode')
    disagg_mode = global_config.get('disagg_mode')

    # Decide on which mode to run based on the run mode

    if run_mode == 'prod' or run_mode == 'custom':

        # Find out if water heater has already been detected by the Disagg run & is a WH disabled pilot

        disable_run = get_detection_status(global_config, item_input_object, hsm_fail, hsm_in, logger_seasonal_wh)

        # If run mode historical or incremental

        if ((disagg_mode == 'historical') or (disagg_mode == 'incremental')) and not disable_run:

            # Calling the seasonal wh module

            item_input_object, item_output_object, exit_status, debug = seasonal_wh_disagg(
                item_input_object, item_output_object, hsm_in, hsm_fail, logger_seasonal_wh_pass)

            if not exit_status:
                pipeline_output_object, item_output_object = update_hsm(pipeline_output_object, item_output_object, debug)

        # If run mode is mtd then use available hsm

        elif disagg_mode == 'mtd' and not hsm_fail and not disable_run:

            # Calling the seasonal wh module

            item_input_object, item_output_object, exit_status, debug = seasonal_wh_disagg(
                item_input_object, item_output_object, hsm_in, hsm_fail, logger_seasonal_wh_pass)

        else:
            if disable_run:
                logger_seasonal_wh.info(
                    'Not running Itemization WH module because WH already detected from Disagg module |')
            else:
                logger_seasonal_wh.info(
                    'Not running Itemization WH module due to HSM fetch failure |')

    else:

        logger_seasonal_wh.error('Unrecognized disagg mode %s |', global_config.get('disagg_mode'))

    t_seasonal_wh_end = datetime.now()
    logger_seasonal_wh.info('The Seasonal WH itemization module took | %.3f s ', get_time_diff(t_seasonal_wh_start,
                                                                                               t_seasonal_wh_end))

    return item_input_object, item_output_object, pipeline_output_object
