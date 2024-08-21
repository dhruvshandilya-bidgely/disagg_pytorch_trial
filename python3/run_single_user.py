"""
Author - Mayank Sharan
Date - 19th Sep 2018
runs the pipeline on a local machine for one user
"""

# Import python packages

import logging
import warnings

from logging.handlers import WatchedFileHandler

# Import functions from within the pipeline

from python3.master_pipeline.master_director import master_director
from python3.initialisation.load_files.fetch_files import fetch_files
from python3.initialisation.load_files.fetch_files import fetch_hybrid_v2_model_files
from python3.initialisation.load_files.load_files_wrapper import load_files_wrapper

# To suppress warnings being printed in the console
warnings.simplefilter('ignore')


def run_single_user(uuid, t_start, t_end, api_env, **kwargs):

    """
    Run the pipeline locally for a single user

    Parameters:
        uuid                    (str)               : The user for which the pipeline has to be run
        t_start                 (int)               : The start timestamp for which the data needs to be pulled
        t_end                   (int)               : The end timestamp for which the data needs to be pulled
        api_env                 (str)               : The api environment the pipeline will be operating in
    """

    # Initialize root logger

    logger_master = logging.getLogger('run_single_user')
    logger_master.setLevel(logging.DEBUG)

    # Set file handler for the log file

    handler = WatchedFileHandler('../gb_disagg_single_user.log')
    formatter = logging.Formatter('%(asctime)s | %(process)d | %(name)s | %(levelname)s | %(uuid)s | %(t_start)d | '
                                  '%(t_end)d | %(pilot_id)d | %(disagg_mode)s | %(message)s')

    handler.setFormatter(formatter)
    logger_master.addHandler(handler)

    logging_dict = {
        'uuid': uuid,
        't_start': -1,
        't_end': -1,
        'pilot_id': -1,
        'disagg_mode': 'N.A.',
    }

    logger_master_adapter = logging.LoggerAdapter(logger_master, logging_dict)

    # Load all static files and models

    logger_pass = {
        'logger': logger_master,
        'logging_dict': logging_dict,
    }

    disagg_version = kwargs.get('disagg_version', '1.0.788')
    job_tag = kwargs.get('job_tag', 'custom')

    fetch_files(api_env, disagg_version, job_tag, logger_master_adapter)
    fetch_hybrid_v2_model_files(api_env, disagg_version, job_tag, logger_master_adapter)

    loaded_files_dict = load_files_wrapper(disagg_version, job_tag, logger_pass)

    # By default we do not write any results in a local run
    # DO NOT CHANGE THIS. Disables write of results

    kwargs['write_results'] = False
    kwargs['smb_type'] = kwargs.get('smb_type', 'all')

    # Initialize cache mode from the parameters. Default value for cache mode is False

    cache_mode = kwargs.get('cache_mode')

    if cache_mode is None:
        cache_mode = False

    # Initialize run mode from the parameters. Default value for run mode is prod

    run_mode = kwargs.get('run_mode')

    if run_mode is None:
        run_mode = 'prod'

    pipeline_mode = kwargs.get('disagg_mode')

    # Fetch data using the api into the disagg input object

    fetch_params = {
        'api_env': api_env,
        'cache_mode': cache_mode,
        'disagg_mode': pipeline_mode,
        'override_mode': "",
        'run_mode': run_mode,
        't_start': t_start,
        't_end': t_end,
        'uuid': uuid,
        'priority': False,
        'cache_update': False,
        'loaded_files': loaded_files_dict,
        'trigger_id': '',
        'trigger_name': '',
        'retry_count': 0,
        'zipcode': '',
        'country': '',
        'build_info': disagg_version.split('.')[-1] + '_' + job_tag
    }

    # Run pipeline for the user with the given parameters

    delete_message = master_director(fetch_params, kwargs, logger_pass)

    if delete_message:
        print('Run successful !')
    else:
        print('Unexpected Issue Encountered')

    # Detach the handler and shut down the logger before completion

    logger_master.removeHandler(handler)
    logging.shutdown()
