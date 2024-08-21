"""
Author - Mayank Sharan
Date - 19/11/18
This function posts the hsms created using the HSM write API
"""

# Import python packages

import time
import json
import logging
import requests

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.mappings.get_env_url import get_post_hsm_url
from python3.utils.oauth_token_manager import OauthTokenManager


def attempt_post_hsm(url, hsm_write_json, header, num_tries, max_retries_primary, max_retries_secondary, logger,
                     server_type):

    """
    Utility function to reduce complexity of the post function
    """

    should_return = False
    is_success = False
    status_code = -1

    # noinspection PyBroadException
    try:
        res = requests.post(url, data=hsm_write_json, headers=header)

        if res.status_code == 201 or res.status_code == 200:

            should_return = True
            is_success = True
            status_code = res.status_code

            logger.info('HSM post attempt successful | try : %d, server : %s', num_tries + 1, server_type)

        elif num_tries == max_retries_primary + max_retries_secondary - 1:
            return True, False, res.status_code
        else:
            logger.info('HSM post attempt failed | try : %d, server : %s, error code : %d', num_tries + 1,
                        server_type, res.status_code)

            time.sleep(Cgbdisagg.API_RETRY_TIME)

    except Exception as exc:

        # Return failure if number of retries are maxed out

        if num_tries == max_retries_primary + max_retries_secondary - 1:
            return True, False, res.status_code

        time.sleep(Cgbdisagg.API_RETRY_TIME)

    return should_return, is_success, status_code


def post_hsm(disagg_input_object, hsm_write_output, logger_pass):

    """
    Parameters:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        hsm_write_output    (list)              : List of all created HSMs in the required format
        logger_pass         (dict)              : Contains logging information needed to log here

    Returns:
        success_status      (bool)              : True if we were able to post else False
    """

    # Initialize logger

    logger_base = logger_pass.get('logger_base').getChild('post_hsm')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Initialise basic variables required
    api_env = disagg_input_object.get('global_config').get('api_env')

    token = OauthTokenManager.get_access_token(api_env)
    uuid = disagg_input_object.get('global_config').get('uuid')

    hsm_write_json = json.dumps(hsm_write_output)

    # Prepare primary URL

    url_raw = get_post_hsm_url(api_env)
    url = url_raw.format(uuid, token)

    # Prepare secondary URL

    url_raw_sec = get_post_hsm_url(api_env, server_type='secondary')
    url_sec = url_raw_sec.format(uuid, token)

    # Log the urls to be used

    logger.info('Primary hsm post url | %s', url)
    logger.info('Secondary hsm post url | %s', url_sec)

    header = {
        'Content-Type': 'application/json',
    }

    # Parameters to post the data in a retry mechanism

    if disagg_input_object.get('global_config').get('priority'):
        max_tries_primary = Cgbdisagg.MAX_TRIES_PRIMARY_API_PRIORITY
    else:
        max_tries_primary = Cgbdisagg.MAX_TRIES_PRIMARY_API

    max_tries_secondary = Cgbdisagg.MAX_TRIES_SECONDARY_API

    # Attempt in a primary secondary server retry structure

    server_type = 'primary'

    for num_tries in range(max_tries_primary + max_tries_secondary):

        # Attempt to post

        should_return, is_success, status_code = attempt_post_hsm(url, hsm_write_json, header, num_tries,
                                                                  max_tries_primary, max_tries_secondary,
                                                                  logger, server_type)

        # As needed return values

        if should_return:
            return is_success, status_code

        # Switch servers if primary fails continually

        if num_tries == max_tries_primary - 1:
            server_type = 'secondary'
            url = url_sec
