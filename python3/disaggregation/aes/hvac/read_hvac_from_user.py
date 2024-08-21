"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to read any pre filled user profile
"""

# Import python packages
import numpy as np


def get_heat_count(app_profile):

    """
    Utility Function to get heating appliance count

    Parameters:

        app_profile (dict)  : Dictionary containing heating appliance count

    Returns:

        app_count   (int)   : Count number for heating
    """

    app_count = -1
    count = []

    # Key description
    # sh - Space Heating
    # rh - Residential Heating
    # ch - Central Heating
    # hp - Heat Pump
    app_name_list = ['sh', 'rh', 'ch', 'hp']

    for app_name in app_name_list:

        app_profile_name = app_profile.get(app_name)

        # if app profle exist, update the count
        if app_profile_name is not None:
            if app_name == 'hp' and app_profile_name.get('number') < 1:
                count.append(0)
            else:
                count.append(app_profile_name.get('number'))

    count = np.array(count)

    pos_idx = count > 0
    neg_idx = count < 0
    zero_idx = count == 0

    # get aggregate heating appliance count
    if np.sum(pos_idx) > 0:
        app_count = np.sum(count[pos_idx])
    elif np.sum(neg_idx) == 0 and np.sum(zero_idx) > 0:
        app_count = 0

    return app_count


def get_cool_count(app_profile):

    """
    Utility Function to get cooling appliance count

    Parameters:

        app_profile (dict)  : Dictionary containing heating appliance count

    Returns:

        app_count   (int)   : Count number for heating
    """

    app_count = -1
    count = []

    # Key description
    # rh - Residential Air Cooling
    # cac - Central Air Cooling
    # hp - Heat Pump
    app_name_list = ['cac', 'rac', 'hp']

    for app_name in app_name_list:

        app_profile_name = app_profile.get(app_name)

        # if app profle exist, update the count
        if app_profile_name is not None:
            if app_name == 'hp' and app_profile_name.get('number') < 1:
                count.append(0)
            else:
                count.append(app_profile_name.get('number'))

    count = np.array(count)

    pos_idx = count > 0
    neg_idx = count < 0
    zero_idx = count == 0

    # get aggregate cooling appliance count
    if np.sum(pos_idx) > 0:
        app_count = np.sum(count[pos_idx])
    elif np.sum(neg_idx) == 0 and np.sum(zero_idx) > 0:
        app_count = 0

    return app_count


def override_detection(disagg_input_object, appliance, base_found):

    """
    Function to override detection if appliance profile says no for AC/SH

    Parameters:

        disagg_input_object (dict)          : Dictionary containing all inputs
        appliance           (str)           : Identifier for AC/SH
        base_found          (bool)          : Boolean indicating if AC/SH is found from HVAC algo in current run

    Returns:

        base_found          (bool)          : Boolean indicating if AC/SH is present as per appliance profile
    """

    app_count = -1

    # get count from app profile
    if appliance == 'sh':
        app_count = get_heat_count(disagg_input_object.get('app_profile'))
    elif appliance == 'ac':
        app_count = get_cool_count(disagg_input_object.get('app_profile'))

    # override base knowledge of appliance found ot not
    if app_count == 0:
        base_found = False
        return base_found
    else:
        return base_found
