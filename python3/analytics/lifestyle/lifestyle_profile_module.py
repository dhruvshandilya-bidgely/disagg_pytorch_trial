"""
Author - Prasoon Patidar
Date - 06th June 2020
Main Lifestyle profile module, calls all relevant billcycle level and seasonal submodules
"""

# import python packages

import logging
from datetime import datetime

# import functions from within the project

from python3.utils.logs_utils import log_prefix
from python3.utils.time.get_time_diff import get_time_diff
from python3.analytics.lifestyle.functions.get_cluster_info import get_daily_clusters
from python3.analytics.lifestyle.functions.run_lifestyle_bc_module import run_lifestyle_bc_module
from python3.analytics.lifestyle.functions.run_lifestyle_event_module import run_lifestyle_event_module
from python3.analytics.lifestyle.functions.run_lifestyle_annual_module import run_lifestyle_annual_module
from python3.analytics.lifestyle.functions.run_lifestyle_season_module import run_lifestyle_season_module


def lifestyle_profile_module(lifestyle_input_object, lifestyle_output_object, logger_pass):

    """
    Parameters:
        lifestyle_input_object (dict)              : Dictionary containing all inputs for lifestyle modules
        lifestyle_output_object(dict)              : Dictionary containing all outputs for lifestyle modules
        logger_pass(dict)                          : Contains base logger and logging dictionary

    Returns:
        lifestyle_output_object(dict)              : Dictionary containing all outputs for lifestyle modules
    """

    t_lifestyle_module_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('lifestyle_module')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Main Lifestyle profile module", log_prefix('Generic'))

    # Get Clustering information for each day

    daily_cluster_vals, day_profile_data = get_daily_clusters(lifestyle_input_object, logger_pass)

    lifestyle_output_object['day_clusters'] = daily_cluster_vals

    lifestyle_output_object['day_profile_data'] = day_profile_data

    # Create Bill Cycle level profile for user

    lifestyle_output_object = run_lifestyle_bc_module(lifestyle_input_object, lifestyle_output_object, logger_pass)

    # Create seasonal level profile for user

    lifestyle_output_object = run_lifestyle_season_module(lifestyle_input_object, lifestyle_output_object, logger_pass)

    # Create Event level profile for user

    lifestyle_output_object = run_lifestyle_event_module(lifestyle_input_object, lifestyle_output_object, logger_pass)

    # Create annual profile for user

    lifestyle_output_object = run_lifestyle_annual_module(lifestyle_input_object, lifestyle_output_object, logger_pass)

    t_lifestyle_module_end = datetime.now()

    logger.info("%s Running main lifestyle module took | %.3f s", log_prefix('Generic'),
                get_time_diff(t_lifestyle_module_start,
                              t_lifestyle_module_end))

    return lifestyle_output_object
