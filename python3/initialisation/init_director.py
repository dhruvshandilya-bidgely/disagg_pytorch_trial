"""
Author - Mayank Sharan
Date - 13/01/19
Fetches all data for the user using cache and returns in a single dictionary
"""

# Import python packages
import logging
import traceback

# Import functions from within the project
from python3.initialisation.fetch_data.fetch_data_api import fetch_data_api
from python3.initialisation.fetch_data.fetch_data_cache import fetch_data_cache


def init_director(fetch_params, run_params_dict, logger_master, logging_dict):
    """
    This function initialises the pipeline_input_objects

    Parameters:
        fetch_params          (dict)              : Contains uuid, t_start, t_end and disagg_mode
        run_params_dict       (dict)              : Dictionary with all custom run parameters provided
        logger_master         (logger)            : The root logger from which to get the child logger
        logging_dict          (dict)              : Dictionary containing all logging parameters

    Returns:
        pipeline_input_object (dict)              : Dictionary containing all inputs to run the pipeline
        delete_message        (bool)              : Boolean containing the status of data fetch
    """

    # Initialise logger pass
    logger_base = logger_master.getChild('init_director')
    logger_pass = {
        'logger': logger_base,
        'logging_dict': logging_dict
    }
    logger = logging.LoggerAdapter(logger_base, logging_dict)

    # Initialise variables required

    cache_mode = fetch_params.get('cache_mode')
    delete_message = True
    pipeline_input_objects = []

    # noinspection PyBroadException
    try:
        # Attempt to pull data if cached
        if cache_mode:
            pipeline_input_objects = fetch_data_cache(fetch_params, run_params_dict, logger_pass)
            # logger_pass   # we have to take a look at some possible bugs

        # In case of cache missing or otherwise load data from API

        if len(pipeline_input_objects) == 0:
            pipeline_input_objects, delete_message = fetch_data_api(fetch_params, run_params_dict, logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Data fetch failed | %s', error_str)
        pipeline_input_objects = None
        delete_message = False

    return pipeline_input_objects, delete_message
