"""
Author - Mayank Sharan
Date - 19/09/18
Fetches data and other parameters based on the queue message
"""

# Import python packages

import time
import logging
import requests
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.mappings.get_env_url import get_data_fetch_url
from python3.utils.oauth_token_manager import OauthTokenManager


def get_pipeline_event(input_data, fetch_params):

    """
    Utility function to create pipeline event for custom mode
    """

    pipeline_mode = fetch_params.get('disagg_mode')

    if pipeline_mode is None:
        pipeline_mode = 'historical'

    pipeline_event = {
        'start': input_data[0, Cgbdisagg.INPUT_EPOCH_IDX],
        'end': input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX],
        "rawDataDurationInDays": int(np.ceil((input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX] -
                                              input_data[0, Cgbdisagg.INPUT_EPOCH_IDX]) /
                                             Cgbdisagg.SEC_IN_DAY)),
        "disaggMode": pipeline_mode,
    }

    return pipeline_event


def attempt_fetch_request_object(url, max_tries, fetch_params, logger, server_type):

    """
    Utility to fetch request object from a given API for a given number of tries
    """

    # Initialize req_object variable

    req_object = {
        'error': {
            'message': 'Non 400 error'
        }
    }

    for num_tries in range(max_tries):

        # noinspection PyBroadException
        try:
            res = requests.get(url)
            req_object = res.json()

            input_data = np.array(req_object['payload']['rawData'])
            input_data = input_data.astype('float')

            pipeline_run_data = req_object['payload']['gbDisaggMetaData']

            weather_analytics_data = req_object['payload']['derivedWeatherData']

            if fetch_params.get('run_mode') == 'custom':
                pipeline_run_data['gbDisaggEvents'] = [get_pipeline_event(input_data, fetch_params)]

            logger.info('Request object fetch attempt successful | try : %d, server : %s', num_tries + 1, server_type)

            return input_data, pipeline_run_data, weather_analytics_data, res.status_code, req_object

        except Exception:

            logger.info('Request object fetch attempt failed | try : %d, server : %s, error code : %d', num_tries + 1,
                        server_type, res.status_code)

            # Return error output if number of retries are maxed out or if the error code is 400

            if num_tries == max_tries - 1 or res.status_code == 400:
                return None, None, None, res.status_code, req_object

            time.sleep(Cgbdisagg.API_RETRY_TIME)

    return None, None, None, -1, req_object


def get_backend_data_fetch_api(fetch_params, logger_pass):

    """
    Parameters:
        fetch_params        (dict)              : Contains api_env, uuid, t_start, t_end and pipeline_mode
        logger_pass         (dict)              : Contains logging information needed to log here

    Returns:
        input_data          (np.array)          : 21 column input data matrix
        pipeline_run_data     (dict)              : Contains information critical to doing the pipeline run
    """

    # Initialize logger

    logger_base = logger_pass.get('logger_base').getChild('fetch_request_object')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Setup the url to pull data from

    token = OauthTokenManager.get_access_token(fetch_params['api_env'])

    # Prepare primary URL

    url_raw = get_data_fetch_url(fetch_params['api_env'])
    url = url_raw.format(fetch_params['uuid'], fetch_params['t_start'], fetch_params['t_end'],
                         fetch_params['override_mode'], token, fetch_params['trigger_id'], fetch_params['trigger_name'],
                         fetch_params['retry_count'])

    # Prepare secondary URL

    url_raw_sec = get_data_fetch_url(fetch_params['api_env'], server_type='secondary')
    url_sec = url_raw_sec.format(fetch_params['uuid'], fetch_params['t_start'], fetch_params['t_end'],
                                 fetch_params['override_mode'], token, fetch_params['trigger_id'], fetch_params['trigger_name'],
                                 fetch_params['retry_count'])

    # Use the custom API for custom mode

    if fetch_params.get('run_mode') == 'custom':
        url += '&onlyRawData=true'
        url_sec += '&onlyRawData=true'

    logger.info('Primary request object fetch url | %s', url)
    logger.info('Secondary request object fetch url | %s', url_sec)

    # Parameters to fetch the data in a retry mechanism

    if fetch_params.get('priority'):
        max_tries_primary = Cgbdisagg.MAX_TRIES_PRIMARY_API_PRIORITY
    else:
        max_tries_primary = Cgbdisagg.MAX_TRIES_PRIMARY_API

    max_tries_secondary = Cgbdisagg.MAX_TRIES_SECONDARY_API

    return url, max_tries_primary, url_sec, max_tries_secondary


