"""
Author - Mayank Sharan
Date - 20/09/18
fetch dictionary of hsm for the appliances that have to be run
"""

# Import python packages

import time
import logging
import requests

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.mappings.get_app_id import get_app_id
from python3.config.mappings.get_env_url import get_static_data_url

from python3.utils.oauth_token_manager import OauthTokenManager


def attempt_fetch_meta_object(url, max_tries, logger, server_type):

    """
    Utility to fetch meta object from a given API for a given number of tries
    """

    for num_tries in range(max_tries):

        # noinspection PyBroadException
        try:
            res = requests.get(url)
            req_object = res.json()

            home_meta_data = req_object['payload']['home']
            app_profile = req_object['payload']['userApplianceProfileDataList']
            hsm_appliances = req_object['payload']['genericHSMGB']

            logger.info('Meta data fetch attempt successful | try : %d, server : %s', num_tries + 1, server_type)

            return home_meta_data, app_profile, hsm_appliances, res.status_code

        except Exception:

            logger.info('Meta data fetch attempt failed | try : %d, server : %s, error code : %d', num_tries + 1,
                        server_type, res.status_code)

            # Return error output if number of retries are maxed out

            if num_tries == max_tries - 1:
                return None, None, None, res.status_code

            time.sleep(Cgbdisagg.API_RETRY_TIME)

    return None, None, None, -1


def fetch_meta_object(fetch_params, logger_pass):

    """
    Parameters:
        fetch_params        (dict)              : Contains all parameters needed to fetch data
        logger_pass         (dict)              : Contains logging information needed to log here

    Output:
        config              (dict)              : Dictionary containing all parameters to run the pipeline
    """

    # Initialize logger

    logger_base = logger_pass.get('logger_base').getChild('fetch_request_object')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Set appliance ids for which we need to pull the HSM

    app_to_pull_hsm_for = Cgbdisagg.APPS_TO_PULL_HSM_FOR
    app_ids_to_pull_hsm_for = []

    for app_name in app_to_pull_hsm_for:
        app_ids_to_pull_hsm_for.append(str(get_app_id(app_name)))

    app_ids_for_api = ','.join(app_ids_to_pull_hsm_for)

    # Setup the url to pull data

    token = OauthTokenManager.get_access_token(fetch_params['api_env'])

    # Prepare primary URL

    url_raw = get_static_data_url(fetch_params['api_env'])
    url = url_raw.format(fetch_params['uuid'], app_ids_for_api, token)

    # Prepare secondary URL

    url_raw_sec = get_static_data_url(fetch_params['api_env'], server_type='secondary')
    url_sec = url_raw_sec.format(fetch_params['uuid'], app_ids_for_api, token)

    logger.info('Primary meta data fetch url | %s', url)
    logger.info('Secondary meta data fetch url | %s', url_sec)

    # Parameters to fetch the data in a retry mechanism

    if fetch_params.get('priority'):
        max_tries_primary = Cgbdisagg.MAX_TRIES_PRIMARY_API_PRIORITY
    else:
        max_tries_primary = Cgbdisagg.MAX_TRIES_PRIMARY_API

    max_tries_secondary = Cgbdisagg.MAX_TRIES_SECONDARY_API

    # Attempt data fetch from primary data server

    home_meta_data, app_profile, hsm_appliances, status_code = attempt_fetch_meta_object(url, max_tries_primary,
                                                                                         logger, server_type='primary')

    if home_meta_data is None:

        # Attempt data fetch from secondary data server

        home_meta_data, app_profile, hsm_appliances, _ = attempt_fetch_meta_object(url_sec, max_tries_secondary,
                                                                                   logger, server_type='secondary')

    return home_meta_data, app_profile, hsm_appliances, status_code
