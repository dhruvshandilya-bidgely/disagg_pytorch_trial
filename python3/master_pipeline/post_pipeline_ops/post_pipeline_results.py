"""
Author - Mayank Sharan
Date - 20/11/18
This function posts the disagg output generated using the HSM write API
"""

# Import python packages

import os
import time
import json
import logging
import requests

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.mappings.get_env_url import get_post_pipeline_url
from python3.utils.oauth_token_manager import OauthTokenManager


def attempt_post_pipeline_output(url, num_tries, pipeline_write_json, header, max_retries_primary, max_retries_secondary,
                                 logger, server_type):

    """
    Utility function to reduce complexity of the post function
    """

    should_return = False
    is_success = False
    status_code = -1

    # noinspection PyBroadException
    try:
        res = requests.post(url, data=pipeline_write_json, headers=header)

        if res.status_code == 201 or res.status_code == 200:

            should_return = True
            is_success = True
            status_code = res.status_code

            logger.info('Pipeline output post attempt successful | try : %d, server : %s', num_tries + 1, server_type)

        elif num_tries == max_retries_primary + max_retries_secondary - 1:
            return True, False, res.status_code
        else:
            logger.info('Pipeline post attempt failed | try : %d, server : %s, error code : %d', num_tries + 1,
                        server_type, res.status_code)

            time.sleep(Cgbdisagg.API_RETRY_TIME)

    except Exception as exc:

        # Return failure if number of retries are maxed out

        if num_tries == max_retries_primary + max_retries_secondary - 1:
            return True, False, res.status_code

        time.sleep(Cgbdisagg.API_RETRY_TIME)

    return should_return, is_success, status_code


def post_pipeline_output(uuid, api_env, pipeline_output, fetch_params, logger_master, logging_dict):

    """
    Parameters:
        uuid                (str)               : Identifier for the user
        api_env             (str)               : Defines the environment in which the API needs to be
        pipeline_output       (list)            : The list of disagg output dictionaries to post using the API
        fetch_params        (dict)              : Dictionary containing the data fetch parameters
        logger_master       (logger)            : Logger object to derive further base logger
        logging_dict        (dict)              : Contains attributes to populate fixed fields in the logger

    Returns:
        success_status      (bool)              : True if we were able to post else False
    """

    # Initialize the logger

    logger_base = logger_master.getChild('post_pipeline_output')
    logger = logging.LoggerAdapter(logger_base, logging_dict)

    # Initialise basic variables required

    token = OauthTokenManager.get_access_token(api_env)

    pipeline_write_json = json.dumps(pipeline_output)
    pipeline_write_json = pipeline_write_json.replace('\"null\"', 'null')

    # Prepare primary URL

    url_raw = get_post_pipeline_url(api_env)
    url = url_raw.format(uuid, token, fetch_params['trigger_id'], fetch_params['trigger_name'], fetch_params['retry_count'])

    # Prepare secondary URL

    url_raw_sec = get_post_pipeline_url(api_env, server_type='secondary')
    url_sec = url_raw_sec.format(uuid, token, fetch_params['trigger_id'], fetch_params['trigger_name'], fetch_params['retry_count'])

    # Log the urls to be used

    logger.info('Primary pipeline output post url | %s', url)
    logger.info('Secondary pipeline output post url | %s', url_sec)

    header = {
        'Content-Type': 'application/json',
    }

    # Parameters to post the data in a retry mechanism

    if fetch_params.get('priority'):
        max_tries_primary = Cgbdisagg.MAX_TRIES_PRIMARY_API_PRIORITY
    else:
        max_tries_primary = Cgbdisagg.MAX_TRIES_PRIMARY_API

    max_tries_secondary = Cgbdisagg.MAX_TRIES_SECONDARY_API

    server_type = 'primary'

    for num_tries in range(max_tries_primary + max_tries_secondary):

        # Attempt to post

        should_return, is_success, status_code = attempt_post_pipeline_output(url, num_tries, pipeline_write_json, header,
                                                                              max_tries_primary, max_tries_secondary,
                                                                              logger, server_type)

        # As needed return values

        if should_return:
            return is_success, status_code

        # Switch servers if primary fails continually

        if num_tries == max_tries_primary - 1:
            server_type = 'secondary'
            url = url_sec
