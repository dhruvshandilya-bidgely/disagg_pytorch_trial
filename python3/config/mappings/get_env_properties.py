"""
Author - Mayank Sharan
Date - 14/01/19
Returns basic properties for a given API env
"""

# Import python packages

import os


def get_env_properties(env):

    """
    Parameters:
        env             (str)               : Environment for which variables need to be extracted
    Returns:
        properties      (dict)              : Dictionary containing basic information
    """

    env_properties = {
        'dev': dict({
            'protocol': 'https://',
            'primary': 'devapi.bidgely.com',
            'secondary': 'devapi.bidgely.com',
            'aws_region': 'us-west-2'
        }),

        'ds': dict({
            'protocol': 'http://',
            'primary': 'dspyapi.bidgely.com',
            'secondary': 'dsapi.bidgely.com',
            'aws_region': 'us-east-1'
        }),

        'nonprodqa': dict({
            'protocol': 'https://',
            'primary': 'nonprodqaapi.bidgely.com',
            'secondary': 'nonprodqaapi.bidgely.com',
            'aws_region': 'us-west-2'
        }),
        'prod-na': dict({
            'protocol': 'https://',
            'primary': 'napyapi.bidgely.com',
            'secondary': 'naapi.bidgely.com',
            'aws_region': 'us-east-1'
        }),
        'prod-eu': dict({
            'protocol': 'https://',
            'primary': 'eupyapi.bidgely.com',
            'secondary': 'euapi.bidgely.com',
            'aws_region': 'eu-central-1'
        }),
        'prod-jp': dict({
            'protocol': 'https://',
            'primary': 'jppyapi.bidgely.com',
            'secondary': 'jpapi.bidgely.com',
            'aws_region': 'ap-northeast-1'
        }),
        'prod-ca': dict({
            'protocol': 'https://',
            'primary': 'capyapi.bidgely.com',
            'secondary': 'caapi.bidgely.com',
            'aws_region': 'ca-central-1'
        }),
        'prod-na-2': dict({
            'protocol': 'https://',
            'primary': 'na2pyapi.bidgely.com',
            'secondary': 'naapi2.bidgely.com',
            'aws_region': 'us-east-1'
        }),
        'preprod-na': dict({
            'protocol': 'https://',
            'primary': 'napreprodapi.bidgely.com',
            'secondary': 'napreprodapi.bidgely.com',
            'aws_region': 'us-east-1'
        }),
        'qaperfenv': dict({
            'protocol': 'http://',
            'primary': 'awseb-e-i-awsebloa-1jk42nlshi8yb-2130246765.us-west-2.elb.amazonaws.com',
            'secondary': 'awseb-e-i-awsebloa-1jk42nlshi8yb-2130246765.us-west-2.elb.amazonaws.com',
            'aws_region': 'us-west-2'
        }),
        'uat': dict({
            'protocol': 'https://',
            'primary': 'uatapi.bidgely.com',
            'secondary': 'uatapi.bidgely.com',
            'aws_region': 'us-west-2'
        }),

        'productqa': dict({
            'protocol': 'https://',
            'primary': 'productqaapi.bidgely.com',
            'secondary': 'productqaapi.bidgely.com',
            'aws_region': 'us-west-2'
        }),
    }

    env_prop = env_properties.get(str.lower(env))

    # Load env variable based properties iff Bidgely env is set

    if os.getenv('BIDGELY_ENV') is not None:

        # Load protocol environment variable if available

        env_protocol = os.getenv("PYAMI_API_PROTOCOL")

        if env_protocol is not None and env_protocol in ['http://', 'https://']:
            env_prop['protocol'] = env_protocol

        # Load primary api server environment variable if available

        env_primary = os.getenv("PYAMI_API_PRIMARY")

        if env_primary is not None:
            env_prop['primary'] = env_primary

        # Load secondary api server environment variable if available

        env_secondary = os.getenv("PYAMI_API_SECONDARY")

        if env_secondary is not None:
            env_prop['secondary'] = env_secondary

    return env_prop
