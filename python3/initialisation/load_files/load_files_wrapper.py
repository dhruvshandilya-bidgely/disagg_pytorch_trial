"""
Author - Mayank Sharan
Date - 12th Aug 2019
Load files wrapper calls different functions that load files for modules and places them in a dictionary to be passed
"""

# Import python packages

import logging
import traceback

# Import functions from within the project

from python3.utils.logs_utils import log_prefix
from python3.initialisation.load_files.load_wh_models import load_wh_files
from python3.initialisation.load_files.load_ev_models import load_ev_files
from python3.initialisation.load_files.load_ref_models import load_ref_files
from python3.initialisation.load_files.load_item_files import load_item_files
from python3.initialisation.load_files.load_solar_models import load_solar_files
from python3.initialisation.load_files.load_lifestyle_models import load_lf_files
from python3.initialisation.load_files.load_ev_propensity_files import load_ev_propensity_files
from python3.initialisation.load_files.load_hvac_ineffficiency_files import load_hvac_inefficiency_files


def load_files_wrapper(disagg_version, job_tag, logger_pass):

    """
    Parameters:
        disagg_version      (string)            : String containing the version information of the build
        job_tag             (string)            : String containing information regarding the build process used
        logger_pass         (dict)              : Contains information needed for logging

    Returns:
        loaded_files        (dict)              : Contains all loaded files
    """

    version = disagg_version.split('.')[-1]

    # Initiate logger for the load files wrapper

    logger_local = logger_pass.get("logger").getChild("load_disagg_files")
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    # Initialize the loaded files dictionary

    loaded_files = {}

    # Call the function to load files to be used by water heater

    # noinspection PyBroadException
    try:
        wh_files = load_wh_files(version, job_tag, logger_pass)
        loaded_files['wh_files'] = wh_files
    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error("Something went wrong while loading files for water heater | %s", error_str)

    # Call the function to load files to be used by EV

    # noinspection PyBroadException
    try:
        ev_files = load_ev_files(version, job_tag, logger_pass)
        loaded_files['ev_files'] = ev_files
    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error("Something went wrong while loading files for ev | %s", error_str)

    # Call the function to load files to be used by EV propensity

    # noinspection PyBroadException
    # try:
    #     ev_propensity_files = load_ev_propensity_files(version, job_tag, logger_pass)
    #     loaded_files['ev_propensity_files'] = ev_propensity_files
    # except Exception:
    #     error_str = (traceback.format_exc()).replace('\n', ' ')
    #     logger.error("Something went wrong while loading files for ev propensity | %s", error_str)

    # Call the function to load files to be used by solar

    # noinspection PyBroadException
    try:
        solar_files = load_solar_files(version, job_tag, logger_pass)
        loaded_files['solar_files'] = solar_files
    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error("Something went wrong while loading files for solar | %s", error_str)

    # Call the function to load files to be used by lifestyle module

    # noinspection PyBroadException
    try:
        lf_files = load_lf_files(version, job_tag, logger_pass)
        loaded_files['lf_files'] = lf_files
    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error("%s Something went wrong while loading files for lifestyle module | %s", log_prefix('Generic'), error_str)

    # noinspection PyBroadException
    try:
        hvac_inefficiency_files = load_hvac_inefficiency_files(version, job_tag, logger_pass)
        loaded_files['hvac_inefficiency_files'] = hvac_inefficiency_files
        logger.info('Successfully loaded hvac inefficiency files |')
    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error("Something went wrong while loading files for HVAC inefficiency | %s", error_str)

    # noinspection PyBroadException
    try:
        ref_files = load_ref_files(version, job_tag, logger_pass)
        loaded_files['ref_files'] = ref_files
    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error("Something went wrong while loading files for hybrid ref module | %s",  error_str)

    # noinspection PyBroadException
    try:
        item_files = load_item_files(version, job_tag, logger_pass)
        loaded_files['item_files'] = item_files
    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error("Something went wrong while loading files for itemization module | %s",  error_str)

    return loaded_files
