
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Cooking config file
"""

# Import python packages

import numpy as np

from python3.itemization.aer.functions.get_config import get_hybrid_config


def get_entertainment_config(samples_per_hour, pilot_config):

    """
    Initialize config required for entertainment ts level estimation module

    Parameters:
        samples_per_hour             (int)           : Total number of samples in an hour
        pilot_config                 (dict)          : hybrid config for the pilot of the given user

    Returns:
        entertainment_config         (dict)           : Dict containing all entertainment related parameters
    """

    hybrid_config = get_hybrid_config(pilot_config)

    ent_idx = hybrid_config.get("app_seq").index('ent')

    scale_cons = hybrid_config.get("scale_app_cons")[ent_idx]
    cons_factor = hybrid_config.get("scale_app_cons_factor")[ent_idx]

    entertainment_config = dict()

    # "usage_hours": entertainment usage hours for different time bands
    # "weekend_present_usage_hours": entertainment usage hours for different time bands (for weekend present usages)
    # "weekend_absent_usage_hours": entertainment usage hours for different time bands (for weekend absent usages)

    general_config = {
        "usage_hours": np.array([1, 3, 3, 3, 3]) * samples_per_hour,
        "weekend_present_usage_hours": [1.5, 2.5, 2.5, 3.5],
        "weekend_absent_usage_hours": [0.5, 1.5, 1, 2]
    }

    # "router_amplitude": router amplitude
    # "tv_ao": tv ao usage amplitude
    # "high_occupants_limit": high consumption for high occupants count users
    # "high_occupants_multiplier": high consumption for high occupants count users

    ao_config = {

        "router_amplitude": 6,
        "tv_ao": 2,
        "high_occupants_limit": 3,
        "high_occupants_multiplier": 2

    }

    # "charging_amp": charging amplitude
    # "devices": charger devices count
    # "default_morning_end": default morning activity end time (for time that will be used to calculating ent usage of the user)
    # "default_mid_day_start": default mid day activity start time (for time that will be used to calculating ent usage of the user)
    # "default_mid_day_end": default mid day activity end time (for time that will be used to calculating ent usage of the user)
    # "default_evening_start": default evening activity start time (for time that will be used to calculating ent usage of the user)
    # "buffer_hours": buffer hours (eg - charging stops before 0.5 hours of leaving for office)
    # "charging_hours": charging hours

    od_low_config = {

        "charging_amp": [10, 150],
        "late_night_hours": 4,
        "devices": [[1, 0],
                    [1, 1],
                    [1, 0],
                    [1, 1],
                    [0, 0]],
        "default_morning_end": 11,
        "default_mid_day_start": 12,
        "default_mid_day_end": 17,
        "default_evening_start": 18,
        "buffer_hours": [0.5, 0, 0, 0.5],
        "charging_hours": 5,
        "zero_usage_hours": np.arange(1*samples_per_hour, 6*samples_per_hour + 1)

    }

    # "television_amp": television amplitude
    # "devices":  charger devices count for different time bands
    # "default_morning_end": default morning activity end time
    # "default_mid_day_start": default mid day activity start time
    # "default_mid_day_end": default mid day activity end time
    # "default_evening_start": default evening activity start time
    # "buffer_hours": buffer hours (eg - tv usage stops before 0.5 hours of leaving for office)

    od_high_config = {

        "television_amp": 400,
        "late_night_hours": 4,
        "devices": [[1],
                    [1],
                    [0],
                    [1],
                    [0]],
        "default_morning_end": 11,
        "default_mid_day_start": 12,
        "default_mid_day_end": 17,
        "default_evening_start": 18,
        "buffer_hours": [0.5, 0, 0, 0.5],
        "zero_usage_hours": np.arange(1*samples_per_hour, 6 * samples_per_hour + 1)

    }

    if scale_cons:
        od_high_config["television_amp"] = od_high_config["television_amp"] * cons_factor

        if cons_factor >= 3:
            od_high_config["buffer_hours"] = np.array(od_high_config["buffer_hours"]) * np.ceil(3)
            od_low_config["buffer_hours"] = np.array(od_low_config["buffer_hours"]) * np.ceil(3)

        elif cons_factor >= 2.2:
            od_high_config["buffer_hours"] = np.array(od_high_config["buffer_hours"]) * np.ceil(2.2)
            od_low_config["buffer_hours"] = np.array(od_low_config["buffer_hours"]) * np.ceil(2.2)

        elif cons_factor >= 1.5:
            od_high_config["buffer_hours"] = np.array(od_high_config["buffer_hours"]) * np.ceil(1.5)
            od_low_config["buffer_hours"] = np.array(od_low_config["buffer_hours"]) * np.ceil(1.5)

    entertainment_config.update({
        "general_config": general_config,
        "od_high_config": od_high_config,
        "od_low_config": od_low_config,
        "ao_config": ao_config
    })

    return entertainment_config
