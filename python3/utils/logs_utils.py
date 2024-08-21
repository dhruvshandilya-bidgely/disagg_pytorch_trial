
"""
Author - Nisha Agarwal
Date - 23rd September 2020
Small utility functions for logging purpose
"""

# import functions from within the project

from python3.analytics.lifestyle.init_lifestyle_config import LifestyleId


def log_prefix(attribute, type='str'):

    """
    Parameters:
        attribute(string)              : attribute in log
        type(string)                   : Type of log prefix
    Returns:
        log_prefix_value(string)       : get relevant input for weekend warrior calculation
    """

    if attribute == 'Generic':
        return attribute +  " |"

    if type == 'list':
        return "lifestyle_id-" + ','.join([str(LifestyleId[xr].value) for xr in attribute]) + " | " \
               + ','.join(attribute) + " |"

    log_prefix_value = "lifestyle_id-" + str(LifestyleId[attribute].value) + " | " + attribute + " |"

    return log_prefix_value
