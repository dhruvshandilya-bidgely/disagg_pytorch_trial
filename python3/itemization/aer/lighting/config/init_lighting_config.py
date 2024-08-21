
"""
Author - Nisha Agarwal
Date - 10th Nov 2020
File containing lighting module config values
"""

# Import python packages

import numpy as np


def init_lighting_config(samples_per_hour):

    """
    Function initializing lighting module config dictionary

    Parameters:
        samples_per_hour         (int)          : number of samples in an hour
    Returns:
        config                   (dict)         : active hours config dict
    """

    config = dict()

    # 'sunrise_val': value used to fill sunrise time of the day ,
    # 'sunset_val': value used to fill sunset time of the day ,
    # 'faulty_days_limit': max number of faulty days required to fill sunrise/sunset days using neighbouring days ,
    # 'default_sunrise': Default sunrise time ,
    # 'default_sunset': Default sunset time

    sunrise_sunset_config = {
        'sunrise_val': -1,
        'sunset_val': 1,
        'faulty_days_limit': 30,
        'default_sunrise': 6,
        'default_sunset': 18
    }

    config.update({

        "sunrise_sunset_config": sunrise_sunset_config

    })

    # 'morn_band_hours': morning lighting band hours
    # 'eve_band_hours': evening lighting band hours
    # 'morn_band_limit': max hours for gap between morning lighting bands
    # 'eve_band_limit': max hours for gap between evening lighting bands

    general_config = {
        'morn_band_hours': np.arange(4*samples_per_hour, 13*samples_per_hour + 1).astype(int),
        'eve_band_hours': np.arange(14*samples_per_hour, 27*samples_per_hour + 1).astype(int),
        'morn_band_limit': 4 * samples_per_hour,
        'eve_band_limit': 4 * samples_per_hour
    }

    config.update({
        "general_config": general_config
    })

    # 'after_sunrise_hours': lighting hours after sunrise ,
    # 'before_sunset_hours': lighting hours before sunset
    # 'morn_buffer_inc': morning buffer hours (hours lighting is given after the sunrise)
    # 'eve_buffer_inc': evening buffer hours (hours lighting is given before the sunset)
    # 'morn_buffer_buc': diff between sunrise and sunset hours bucket - used to determine morning buffer hours
    # 'eve_buffer_buc': diff between sunrise and sunset hours bucket - used to determine evening buffer hours

    usage_potential_config = {
        'after_sunrise_hours': 2,
        'before_sunset_hours': 1,
        'morn_buffer_inc': [0, 0.5, 1],
        'eve_buffer_inc': [0, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1.25, 1.25, 1.25, 1.25, 1.25],
        'morn_buffer_buc': [15*samples_per_hour, 16*samples_per_hour],
        'eve_buffer_buc': [14.5 * samples_per_hour, 14.75 * samples_per_hour, 15 * samples_per_hour,
                           15.25 * samples_per_hour, 15.5 * samples_per_hour, 16 * samples_per_hour,
                           16.25 * samples_per_hour, 16.5 * samples_per_hour, 16.75 * samples_per_hour,
                           17 * samples_per_hour, 17.25 * samples_per_hour, 17.5 * samples_per_hour,
                           17.75 * samples_per_hour]
    }

    config.update({
        "usage_potential_config": usage_potential_config
    })

    # 'score_limit': score limits to calculate top cleanest days
    # 'days_count': days limits to calculate top cleanest days ,
    # 'default_days_count': default number of top cleanest days, for users with lower cleanliness scores
    # 'min_days': consider all days in case of increamental/mtd/less days data users

    top_clean_days_config = {
        'score_limit': [0.8, 0.7, 0.6],
        'days_count': [10, 30, 20],
        'default_days_count': 10,
        'min_days': 30
    }

    config.update({

        "top_clean_days_config": top_clean_days_config

    })

    # 'score_limit': distribution used to estimate whether to remove min non zero value (baseload) from the input data
    # 'days_fraction'

    remove_min_config = {
        'score_limit': [0.4, 0.6, 0.6, 0.7],
        'days_fraction': [0.7, 0.25, 0.4, 0.3]
    }

    # Special handling for low consumption homes, or clean usage home (in EU and Asia)

    config.update({

        "remove_min_config": remove_min_config

    })

    app_profile_config = {
        'lighting_app_list': [71]
    }

    # 'lighting_app_list': list of lighting appliance profiles

    config.update({

        "app_profile_config": app_profile_config

    })

    # 'tou_percentile_cap': top percentile cap value while smoothing,
    # 'final_percentile_cap': final usage potential percentile cap value ,
    # 'smoothing_window': window used for smoothing using neighbouring points

    smooth_estimate_config = {
        'tou_percentile_cap': 70,
        'final_percentile_cap': 93,
        'smoothing_window': 3

    }

    config.update({

        "smooth_estimate_config": smooth_estimate_config

    })

    # 'multiplier': Parameters for decreasing lighting estimate away from sunrise/sunset hours
    # 'decreament_factor':
    # 'non_lighting_hours': hours with zero lighting

    postprocess_lighting_config = {
        'multiplier': 0.7,
        'decreament_factor': 0.2,
        'non_lighting_hours': [11, 13]

    }

    config.update({

        "postprocess_lighting_config": postprocess_lighting_config

    })

    morning = np.ones(700) * 50
    evening = np.ones(700) * 50

    # The fraction of lighting capacity decreases with activity delta
    # This is because with higher energy delta, the user might be using higher number of living load appliances,
    # and lighting will have a lower contribution to the total energy

    morning[np.arange(0, 15).astype(int)] = (np.arange(0, 15).astype(int) * 10) * 0.5
    morning[np.arange(15, 30).astype(int)] = (np.arange(15, 30).astype(int) * 10) * 0.4
    morning[np.arange(30, 60).astype(int)] = (np.arange(30, 60).astype(int) * 10) * 0.35
    morning[np.arange(60, 700).astype(int)] = (np.arange(60, 700).astype(int) * 10) * 0.3

    evening[np.arange(0, 30).astype(int)] = (np.arange(0, 30).astype(int) * 10) * 0.5
    evening[np.arange(30, 45).astype(int)] = (np.arange(30, 45).astype(int) * 10) * 0.3
    evening[np.arange(45, 60).astype(int)] = (np.arange(45, 60).astype(int) * 10) * 0.25
    evening[np.arange(60, 700).astype(int)] = (np.arange(60, 700).astype(int) * 10) * 0.2

    # 'morning_val': relation between morning delta and morning lighting capacity,
    # 'evening_val': relation between evening delta and evening lighting capacity,
    # 'max_limit': max lighting capacity,
    # 'min_limit': min lighting capacity,
    # 'morning_estimate_weightage': weightage given to morning estimate,
    # 'evening_estimate_weightage': weightage given to evening estimate,
    # 'occupants_bucket': number of occupants buckets ,
    # 'occupants_multiplier': multipler for number of occupants,
    # 'rooms_bucket': number of rooms buckets ,
    # 'rooms_multiplier': multiplier using number of rooms ,
    # 'levels_bucket': number of levels buckets ,
    # 'levels_multiplier': multiplier using number of number ,
    # 'multiplier_factor': multiplication factor used to determine capacity for various sampling rates ,
    # 'last_bucket': minimum fraction of lighting consumption for high consumption users

    lighting_capacity_config = {
        'morning_val': morning,
        'evening_val': evening,
        'max_limit': 600,
        'min_limit': 60,
        'morning_estimate_weightage': 0.35,
        'evening_estimate_weightage': 0.65,
        'occupants_bucket': [1, 2, 5, 9],
        'occupants_multiplier': [-10, 0, 5, 10],
        'rooms_bucket': [1, 1, 3, 9],
        'rooms_multiplier': [-10, 0, 5, 10],
        'levels_bucket': [2, 3, 5, 9],
        'levels_multiplier': [-10, 0, 5, 10],
        'multiplier_factor': 1.3,
        'last_bucket': 0.2
    }

    config.update({

        "lighting_capacity_config": lighting_capacity_config

    })

    # 'consumption_perc': percentile taken to calculate cleanest days living load ,
    # 'morning_start': morning start of activity,
    # 'after_sunrise_hours': morning activity after sunrise,
    # 'evening_end': End of night activity ,
    # 'before_sunset_hours': start of activity before sunset,
    # 'safety_check_max_limit': Parameters to avoid underestimation cases ,
    # 'safety_check_multiplier': Parameters used to calculate minimum amount of required living load
    # 'safety_check_fraction':

    living_load_config = {
        'consumption_perc': 70,
        'morning_start': 4,
        'after_sunrise_hours': 3,
        'evening_end': 2,
        'before_sunset_hours': 1,
        'safety_check_max_limit': 300,
        'safety_check_multiplier': 1.3,
        'safety_check_fraction': 0.75
    }

    config.update({

        "living_load_config": living_load_config

    })

    estimation_config = {
        'default_cap': 200,
        'default_morn': np.arange(8*samples_per_hour, 10*samples_per_hour + 1),
        'default_eve': np.arange(16*samples_per_hour, 21*samples_per_hour + 1),
        'random_arr_buc': [0.3, 0.8]
    }

    config.update({

        "estimation_config": estimation_config

    })

    return config
