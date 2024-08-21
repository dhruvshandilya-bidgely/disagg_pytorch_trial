
"""
Author - Nisha Agarwal
Date - 9th Feb 20
laundry estimation config file
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.functions.get_config import get_hybrid_config


def get_estimation_config(pilot_level_config, samples_per_hour, drier_present=1):

    """
    Initialize config required for laundry estimation module

    Parameters:
        pilot_level_config           (dict)          : hybrid config for the pilot
        samples_per_hour             (int)           : Total number of samples in an hour
        drier_present                (int)           : Flag that represents whether drier is present for this user

    Returns:
        estimation_config            (dict)          : Dict containing all laundry estimation related parameters
    """

    estimation_config = dict()

    # "delta_arr": living load delta buckets to identify presence of dishwasher,
    # "devices_arr": dishwasher count buckets ,
    # "amp_arr": dishwasher amplitude buckets ,
    # "hours_arr": dishwasher duration buckets,
    # "dow_arr": dishwasher weekly frequency buckets,
    # "default_dow": default weekly count ,
    # "default_hours": default dishwasher duration

    dishwasher_config = dict({

        "delta_arr": [1500, 2500],
        "devices_arr": [0, 1, 1],
        "amp_arr": [0, 1500, 1800],
        "hours_arr": [0, int(1.5 * samples_per_hour), int(1.5 * samples_per_hour)],
        "dow_arr": [5, 5, 5],
        "night_hours": np.arange(19 * samples_per_hour, Cgbdisagg.HRS_IN_DAY * samples_per_hour).astype(int),
        "default_dow": 5,
        "default_hours": int(2 * samples_per_hour)

    })

    # modify dishwasher threshold based on min/max/avg config

    hybrid_config = get_hybrid_config(pilot_level_config)

    app = 'ld'

    if hybrid_config.get("dishwash_cat") == "cook":
        app = 'cook'

    # modify laundry threshold based on min/max/avg hybrid config

    ld_idx = hybrid_config.get("app_seq").index(app)

    scale_cons = hybrid_config.get("scale_app_cons")[ld_idx]
    cons_factor = hybrid_config.get("scale_app_cons_factor")[ld_idx]

    if scale_cons:
        dishwasher_config["amp_arr"] = np.array(dishwasher_config["amp_arr"]) * cons_factor

    if cons_factor > 1.5:
        dishwasher_config["delta_arr"] = np.array(dishwasher_config["delta_arr"]) - 1000
        dishwasher_config["hours_arr"] = [int(1 * samples_per_hour), int(1.5 * samples_per_hour), int(1.5 * samples_per_hour)]

    elif cons_factor > 1.3:
        dishwasher_config["delta_arr"] = np.array(dishwasher_config["delta_arr"]) - 500
        dishwasher_config["hours_arr"] = [int(1 * samples_per_hour), int(1.5 * samples_per_hour), int(1.5 * samples_per_hour)]

    estimation_config.update({
        "dishwasher_config": dishwasher_config
    })

    hybrid_config = get_hybrid_config(pilot_level_config)

    ld_idx = hybrid_config.get("app_seq").index('ld')

    scale_cons = hybrid_config.get("scale_app_cons")[ld_idx]
    cons_factor = hybrid_config.get("scale_app_cons_factor")[ld_idx]

    # "delta_arr": laundry delta buckets to identify presence of dishwasher,
    # "devices_arr": laundry count buckets ,
    # "amp_arr": laundry amplitude buckets ,
    # "hours_arr": laundry duration buckets,
    # "dow_arr": laundry weekly frequency buckets,
    # "default_dow": default weekly count ,
    # "default_hours": default laundry duration
    # "default_hours": default laundry count

    washing_machine_config = dict({

        "delta_arr": [100, 200, 400, 1000, 1500],
        "devices_arr": [1, 1, 1, 1, 1, 1],
        "amp_arr": [400, 700, 850, 1000, 1300, 1500],
        "dow_arr": [4, 5, 7, 7, 7, 7],
        "hours_arr": [2*samples_per_hour, 2*samples_per_hour, 2*samples_per_hour, 2*samples_per_hour,
                      2*samples_per_hour, max(2, 2 * samples_per_hour), max(2, 2 * samples_per_hour)],
        "default_dow": 7,
        "default_amp": 1200,
        "default_hours": int(2 * samples_per_hour),
        "default_devices": 1,
        "energy_diff_limit": 600
    })

    # modify laundry threshold based on min/max/avg hybrid config

    if scale_cons:
        washing_machine_config["amp_arr"] = np.array(washing_machine_config["amp_arr"]) * cons_factor
        washing_machine_config["default_amp"] = (washing_machine_config["default_amp"]) * cons_factor

    if cons_factor > 1.35:
        washing_machine_config["hours_arr"] = np.array(washing_machine_config["hours_arr"]) + samples_per_hour
        washing_machine_config["devices_arr"] = np.array(washing_machine_config["devices_arr"]) + 1

    if ("6" in pilot_level_config.get('ld_config').get("drop_app")) or drier_present == 1:
        washing_machine_config["default_hours"] = int(washing_machine_config["default_hours"] / 1.5)
        washing_machine_config["hours_arr"] = (np.array(washing_machine_config["hours_arr"]) / 1.5).astype(int)

    estimation_config.update({
        "washing_machine_config": washing_machine_config
    })

    estimation_config.update({

        'breakfast_hours': np.arange(6 * samples_per_hour, 9.5 * samples_per_hour + 1).astype(int),
        'lunch_hours': np.arange(12 * samples_per_hour, 13 * samples_per_hour + 1).astype(int),
        'dinner_hours': np.arange(18 * samples_per_hour, 21 * samples_per_hour + 1).astype(int),
        'default_max_val': 0,
        'default_min_val': 1000000,
        'cooking_offset': [100, 300, 300],
        'non_laundry_hours': np.arange(1, 6 * samples_per_hour + 1).astype(int),
        'zero_laundry_hours': np.arange((23) * samples_per_hour, 24 * samples_per_hour).astype(int),

        'zero_laundry_hours_low_cons_pilots': np.append(np.arange(0, 6 * samples_per_hour + 1),
                                                        np.arange(21 * samples_per_hour, 24 * samples_per_hour)).astype(int),

        'office_going_hours': np.arange(11 * samples_per_hour, 17 * samples_per_hour + 1).astype(int),
        'weekend_weekday_energy_diff_limit': 250
    })

    return estimation_config
