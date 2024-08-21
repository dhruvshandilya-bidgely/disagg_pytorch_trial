"""
Author - Prasoon Patidar
Date - 3rd June 2020
lifestyle profile module wrapper
"""

# import python packages

import os
import json
import copy
import logging
import traceback
import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.logs_utils import log_prefix

from python3.analytics.lifestyle.lifestyle_profile_module import lifestyle_profile_module
from python3.analytics.lifestyle.functions.plot_lifestyle_heatmaps import plot_lifestyle_heatmaps
from python3.analytics.lifestyle.functions.init_lifestyle_input_object import init_lifestyle_input_object
from python3.analytics.lifestyle.functions.lifestyle_utils import populate_lifestyle_hsm
from python3.analytics.lifestyle.functions.populate_lifestyle_user_profile import populate_lifestyle_user_profile


def lifestyle_profile_wrapper(disagg_input_object, disagg_output_object):

    """
    Parameters:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs

    Returns:
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # initiate logger for lifestyle module

    #TODO: (Nisha) Read input data from disagg input object only once.

    logger_lifestyle_base = disagg_input_object.get('logger').getChild('lifestyle_profile_wrapper')
    logger_lifestyle = logging.LoggerAdapter(logger_lifestyle_base, disagg_input_object.get('logging_dict'))
    logger_lifestyle_pass = {
        'logger_base': logger_lifestyle_base,
        'logging_dict': disagg_input_object.get('logging_dict'),
    }

    if np.all(disagg_input_object["input_data"][:, Cgbdisagg.INPUT_CONSUMPTION_IDX] == 0):

        logger_lifestyle.info("%s Not running lifestyle module since all consumption values are zero", log_prefix('Generic'))

        return disagg_output_object

    # If model fetching is failed, return without processing lifestyle module

    all_lifestyle_models = disagg_input_object.get('loaded_files').get('lf_files')

    run_lifestyle_module_status = all_lifestyle_models.get('run_lifestyle_module_status')

    if run_lifestyle_module_status is False:

        logger_lifestyle.error("%s Required lifestyle models are not fetched from static files, not running lifestyle module",
                               log_prefix('Generic'))

        return disagg_output_object

    # initialize global lifestyle input object to store input requirements for each step

    lifestyle_input_object = init_lifestyle_input_object(disagg_input_object, disagg_output_object,
                                                         logger_lifestyle_pass)

    # initialize global lifestyle output object to store output for each processing step

    lifestyle_output_object = dict()

    # Call Main lifestyle module. Code can branch here for various pipeline level settings

    disagg_mode = disagg_input_object.get('config').get('disagg_mode')

    if not (disagg_mode == 'mtd'):

        lifestyle_output_object = lifestyle_profile_module(lifestyle_input_object, lifestyle_output_object,
                                                           logger_lifestyle_pass)

        # Fill lifestyle profile in disagg output object

        disagg_output_object = populate_lifestyle_user_profile(lifestyle_input_object, lifestyle_output_object,
                                                               disagg_output_object, logger_lifestyle_pass)

        disagg_output_object = populate_lifestyle_hsm(lifestyle_output_object, disagg_input_object, disagg_output_object)

        debug_config = lifestyle_input_object.get('debug_config')

        is_debug = debug_config.get('debug_mode')

        if is_debug:

            # Plot heatmaps for non-dev QA

            try:

                plot_lifestyle_heatmaps(lifestyle_input_object, lifestyle_output_object,
                                        disagg_input_object, disagg_output_object, logger_lifestyle_pass)
            except Exception:

                error_str = (traceback.format_exc()).replace('\n', ' ')

                logger_lifestyle.warning("%s Unable to dump debug plots due to error: %s",
                                         log_prefix('Generic'), error_str)

            # Dump Resulting lifestyle profile for non-dev QA

            try:
                lifestyle_profile = disagg_output_object.get('lifestyle_profile')

                uuid = disagg_input_object.get('config').get('uuid')

                dump_dir = debug_config.get('result_dump_dir')

                disagg_mode = disagg_input_object.get('config').get('disagg_mode')

                dump_dir = dump_dir + '/' + str(disagg_mode)

                if not os.path.exists(dump_dir):
                    os.makedirs(dump_dir)

                dump_file = dump_dir + "/" + uuid + ".json"

                json.dump(lifestyle_profile, open(dump_file, 'w'))

            except Exception:

                error_str = (traceback.format_exc()).replace('\n', ' ')

                logger_lifestyle.warning("%s Unable to dump lifestyle profile due to error: %s",
                                         log_prefix('Generic'), error_str)

    else:

        logger_lifestyle.info("%s Not running lifestyle module in disagg mode mtd", log_prefix('Generic'))

    disagg_output_object['lifestyle_season'] = lifestyle_output_object.get('season')

    return disagg_output_object
