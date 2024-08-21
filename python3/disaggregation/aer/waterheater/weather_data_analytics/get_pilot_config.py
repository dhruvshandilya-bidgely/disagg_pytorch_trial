"""
Author: Mayank Sharan
Created: 24-Mar-2020
Returns pilot specific configurations
"""

# Import project functions and classes

from python3.disaggregation.aer.waterheater.weather_data_analytics.nbi_constants import ConfigConstants


def get_pilot_config(pilot_id):

    """
    Returns pilot specific config values for a given pilot id
    Parameters:
        pilot_id        (int)               : Pilot id for which variables need to be extracted
    Returns:
        config_dict     (dict)              : Dictionary containing basic information
    """

    pilot_config = dict({})

    # Add challenge config for BCH, pilot id 10035

    pilot_config[10035] = {
        'chlg_target_pct': 10,
    }

    # Add global config

    pilot_config[ConfigConstants.default_pilot_id] = {
        'peer_perc_thresh': 60,
        'self_perc_thresh': 90,
    }

    config_dict = pilot_config.get(pilot_id, {})

    return config_dict
