"""
Author - Mayank Sharan
Date - 20/09/18
Returns url format based on the API env
"""

# Import functions from within the project

from python3.config.mappings.get_env_properties import get_env_properties


def get_base_url(env, server_type='primary'):

    """
    Parameters:
        env                 (string)            : The API environment in which pipeline is being executed
        server_type         (string)            : The base_url to hit primary or secondary

    Returns:
        base_url            (string)            : Base url for all API calls based on the API environment
        protocol            (string)            : Protocol configured for the environment http or https
    """

    env_prop = get_env_properties(env)

    base_url = env_prop.get(server_type)
    protocol = env_prop.get('protocol')

    return base_url, protocol


def get_data_fetch_url(env, server_type='primary'):

    """
    Parameters:
        env                 (string)            : The API environment in which pipeline is being executed
        server_type         (string)            : The base_url to hit primary or secondary

    Returns:
        url_format          (string)            : Url format for the specific call in the given environment
    """

    # Get the base url and protocol

    base_url, protocol = get_base_url(env, server_type)

    # Prepare raw url with base url and return

    url_format = protocol + base_url + '/2.1/gb-disagg/process-request/{0}/1?start={1}&end={2}&overrideDisaggMode={3}&' \
                                       'access_token={4}&triggerId={5}&triggerName={6}&retryCount={7}'

    return url_format


def get_static_data_url(env, server_type='primary'):

    """
    Parameters:
        env                 (string)            : The API environment in which pipeline is being executed
        server_type         (string)            : The base_url to hit primary or secondary

    Returns:
        url_format          (string)            : Url format for the specific call in the given environment
    """

    # Get the base url and protocol

    base_url, protocol = get_base_url(env, server_type)

    # Prepare raw url with base url and return

    url_format = protocol + base_url + '/2.1/gb-disagg/user-profile/{0}/1?hsm_appliances={1}&access_token={2}'

    return url_format


def get_post_hsm_url(env, server_type='primary'):

    """
    Parameters:
        env                 (string)            : The API environment in which pipeline is being executed
        server_type         (string)            : The base_url to hit primary or secondary

    Returns:
        url_format          (string)            : Url format for the specific call in the given environment
    """

    # Get the base url and protocol

    base_url, protocol = get_base_url(env, server_type)

    # Prepare raw url with base url and return

    url_format = protocol + base_url + '/2.1/gb-disagg/gb-generic-hsm/{0}/1?access_token={1}'

    return url_format


def get_post_pipeline_url(env, server_type='primary'):

    """
    Parameters:
        env                 (string)            : The API environment in which pipeline is being executed
        server_type         (string)            : The base_url to hit primary or secondary

    Returns:
        url_format          (string)            : Url format for the specific call in the given environment
    """

    # Get the base url and protocol

    base_url, protocol = get_base_url(env, server_type)

    # Prepare raw url with base url and return

    url_format = protocol + base_url + '/2.1/gb-disagg/gb-output/{0}/1?access_token={1}&triggerId={2}&triggerName={3}&retryCount={4}'

    return url_format
