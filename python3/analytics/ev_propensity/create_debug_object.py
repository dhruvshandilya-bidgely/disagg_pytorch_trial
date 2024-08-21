"""
Author - Paras Tehria
Date - 27 May 2021
Call the create_debug_object function to create and initialize ev propensity debug object
"""

import numpy as np

from copy import deepcopy

lifestyle_features_to_extract = ['OfficeGoer', 'ActiveUser', 'DailyLoadType']


def get_lifestyle_attributes(analytics_output_object):
    """
    Get Lifestyle attributes from analytics output object
    Parameters:
        analytics_output_object              (dict)        : Dictionary containing information about analytics outputs
    Returns:
        lifestyle_attributes                 (dict)        : Dict containing lifestyle attributes
    """

    lifestyle_profile = deepcopy(analytics_output_object.get('lifestyle_profile'))

    lifestyle_attributes = dict()

    if lifestyle_profile is not None and len(lifestyle_profile) > 1:

        last_key = list(lifestyle_profile.keys())[-1]
        lifestyle_profile = lifestyle_profile[last_key].get('profileList')[-1]

        for key, value in lifestyle_profile.items():
            if "lifestyleid" in key and value['name'] in lifestyle_features_to_extract:
                lifestyle_attributes[value['name']] = value['value']

    return lifestyle_attributes


def get_solar_status(analytics_input_object, analytics_output_object):
    """
    Checks if the users is a solar user or not
    Parameters:
        analytics_input_object               (dict)        : Dictionary containing information about analytics inputs
        analytics_output_object              (dict)        : Dictionary containing information about analytics outputs
    Returns:
        solar_present                        (dict)        : 1 if solar was detected of user said yes to solar, 0 otherwise
    """

    # Using Python coding standard EAFP. (Easier to ask for forgiveness than permission)
    try:
        solar_present = int(analytics_output_object.get('created_hsm', {}).get('solar', {}).get('attributes', {}).get(
            'solar_present', 0))
    except (AttributeError, KeyError):
        solar_present = 0

    try:
        solar_present = solar_present or (
            analytics_input_object.get('app_profile', {}).get('solar', {}).get('number', 0) > 0)
    except (AttributeError, KeyError):
        solar_present = solar_present

    return int(solar_present)


def create_debug_object(analytics_input_object, analytics_output_object):
    """
    Create debug object for EV propensity module
    Parameters:
        analytics_output_object              (dict)        : Dictionary containing information about analytics outputs
        analytics_input_object               (dict)        : Dictionary containing information about analytics inputs
    Returns:
        debug                                   (dict)     : Created debug object for EV propensity module
    """

    debug = {
        'input_data': deepcopy(analytics_input_object.get('input_data')),
        'propensity_model': analytics_input_object.get('loaded_files', {}).get('ev_propensity_files', {}).get('model'),

        'zipcode_db_file_path': analytics_input_object.get('loaded_files', {}).get('ev_propensity_files', {}).get('zipcode_db_file_path'),

        'na_charging_station_data': analytics_input_object.get('loaded_files', {}).get('ev_propensity_files', {}).get(
            'charging_station_data'),

        'solar_present': get_solar_status(analytics_output_object, analytics_input_object),

        'lifestyle_attributes': get_lifestyle_attributes(analytics_output_object),
        'dwelling': analytics_input_object.get('home_meta_data', {}).get('dwelling'),
        'ownershiptype': analytics_input_object.get('home_meta_data', {}).get('ownershipType'),
    }

    return debug
