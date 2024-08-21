
"""
Author - Nisha Agarwal
Date - 10th Mar 2023
prepare stat app config using given hybrid v2 pilot config file
"""

# Import python packages

import numpy as np


def get_hybrid_config(pilot_config=None):

    """
    Initialize config required for hybrid v2 estimation

    Parameters:
        pilot_level             (dict)           : Dict containing all hybrid v2 pilot config
    Returns:
        config                  (dict)           : prepared config dict
    """

    app = ["cook", "ent", "ld"]

    if pilot_config is None:
        coverage = 90
        dishwash_cat = 'ld'

        default = np.array([25, 35, 40])

        mid_cons = np.array([-1, -1, -1])
        max_cons = np.array([200, 250, 300])
        min_cons = np.array([5, 5, 5])
    else:
        coverage = pilot_config.get('ld_config').get('coverage')
        dishwash_cat = pilot_config.get('dish_washer_cat')

        default = np.array([25, 35, 40])

        default = default - 5 * (pilot_config.get('geography') == "eu")

        mid_cons = np.array([pilot_config.get('cook_config').get('bounds').get('mid_cons'),
                             pilot_config.get('ent_config').get('bounds').get('mid_cons'),
                             pilot_config.get('ld_config').get('bounds').get('mid_cons')])
        max_cons = np.array([pilot_config.get('cook_config').get('bounds').get('max_cons'),
                             pilot_config.get('ent_config').get('bounds').get('max_cons'),
                             pilot_config.get('ld_config').get('bounds').get('max_cons')])
        min_cons = np.array([pilot_config.get('cook_config').get('bounds').get('min_cons'),
                             pilot_config.get('ent_config').get('bounds').get('min_cons'),
                             pilot_config.get('ld_config').get('bounds').get('min_cons')])

    have_mid_cons = mid_cons != -1
    have_max_cons = max_cons > 0
    have_min_cons = min_cons > 0

    scale_ts_level_lim = np.abs(default - mid_cons) > 10

    ts_level_lim_factor = np.fmax(0.5, np.divide(default, mid_cons))
    ts_level_lim_factor = np.fmin(2, ts_level_lim_factor)

    scale_app_cons = np.abs(default - mid_cons) > 5

    scale_app_cons_factor = np.fmax(0.3, np.divide(mid_cons, default))
    scale_app_cons_factor = np.fmin(3, scale_app_cons_factor)

    scale_app_cons[mid_cons <= 0] = 0
    scale_ts_level_lim[mid_cons <= 0] = 0

    min_ts_level_lim = np.abs(default - mid_cons) < -10
    max_ts_level_lim = np.abs(default - mid_cons) > 10

    hard_min_lim = min_cons * 0.8
    have_hard_min_lim = min_cons > 0

    hard_max_lim = max_cons * 1.1
    have_hard_max_lim = max_cons > 0

    change_app_seq = 0
    new_app_seq = []

    if mid_cons[0] > 35 and mid_cons[2] < 20:
        change_app_seq = 1
        new_app_seq = ["cook", 'ent', 'ld']

    elif (mid_cons[0] > 35 and mid_cons[2] < 40) or (mid_cons[0] > 60 and mid_cons[2] < 50):
        change_app_seq = 1
        new_app_seq = ["cook", 'ld', 'ent']

    elif mid_cons[0] > 35 and mid_cons[1] < 30:
        change_app_seq = 1
        new_app_seq = ["ld", 'cook', 'ent']

    elif mid_cons[1] > 60 and mid_cons[2] < 20 and mid_cons[0] > 30:
        change_app_seq = 1
        new_app_seq = ["ent", 'cook', 'ld']

    change_box_detection = np.abs(default - mid_cons) > 10

    box_det_max_cons_limit = np.array([0, 0, 0])
    box_det_min_cons_limit = np.array([0, 0, 0])
    box_det_max_len_limit = np.array([0, 0, 0])
    box_det_min_len_limit = np.array([0, 0, 0])

    if mid_cons[2] > 60:
        box_det_max_cons_limit[2] = np.fmin(5000, (mid_cons[2] /  60) * 3000)

    if mid_cons[2] < 40:
        box_det_max_cons_limit[2] = np.fmax(500, (mid_cons[2] /  40) * 3000)

    if mid_cons[0] > 35:
        box_det_max_cons_limit[0] = np.fmin(5000, (mid_cons[0] / 35) * 4000)

    if mid_cons[0] < 20:
        box_det_max_cons_limit[0] = np.fmax(800, (mid_cons[0] / 20) * 4000)

    box_det_max_cons_limit[mid_cons <= 0] = 0

    if dishwash_cat == "cook":
        change_box_detection[2] = 1
        box_det_max_cons_limit[2] = 2000

    cook_season = np.zeros(12)
    ld_season = np.zeros(12)
    wh_season = np.zeros(12)
    ref_season = np.zeros(12)
    ent_season = np.zeros(12)
    li_season = np.zeros(12)
    cooling_season = np.zeros(12)
    heating_season = np.zeros(12)

    ld_season[:] = 0
    ent_season[:] = 0
    cook_season[:] = 0

    ref_season[:] = [-0.1,-0.05, -0.02, 0, 0.05, 0.07, 0.1, 0.1, 0.05, 0, -0.05, -0.1]
    wh_season[:] = [0.2, 0.2, 0.1,0.0,-0.1,-0.2, -0.2,-0.2, -0.2, 0.0, 0.1,0.2]

    li_season[:] = [-0.1,-0.05, -0.02, 0, 0.05, 0.07, 0.1, 0.1, 0.05, 0, -0.05, -0.1]
    cooling_season[:] = [0.2, 0.2, 0.1,0.0,-0.1,-0.2, -0.2,-0.2, -0.2, 0.0, 0.1,0.2]
    heating_season[:] = [0.2, 0.2, 0.1,0.0,-0.1,-0.2, -0.2,-0.2, -0.2, 0.0, 0.1,0.2]

    config = dict(

        {
            "app_seq": app,

            "mid_cons": mid_cons,
            "max_cons": max_cons,
            "min_cons": min_cons,

            "have_mid_cons": have_mid_cons,
            "have_min_cons": have_min_cons,
            "have_max_cons": have_max_cons,

            "scale_ts_level_lim": scale_ts_level_lim,
            "ts_level_lim_factor": ts_level_lim_factor,
            "max_ts_level_lim": max_ts_level_lim,
            "min_ts_level_lim": min_ts_level_lim,

            "scale_app_cons":scale_app_cons,
            "scale_app_cons_factor": scale_app_cons_factor,

            "hard_min_lim": hard_min_lim,
            "have_hard_min_lim": have_hard_min_lim,

            "hard_max_lim": hard_max_lim,
            "have_hard_max_lim": have_hard_max_lim,

            "change_app_seq": change_app_seq,
            "new_app_seq": new_app_seq,

            "change_box_detection": change_box_detection,

            "box_det_max_cons_limit": box_det_max_cons_limit,
            "box_det_min_cons_limit": box_det_min_cons_limit,
            "box_det_max_len_limit": box_det_max_len_limit,
            "box_det_min_len_limit": box_det_min_len_limit,

            "coverage": coverage,
            "dishwash_cat": dishwash_cat,
            "geography": pilot_config.get('geography'),

            "cook_season": cook_season,
            "ld_season": ld_season,
            "ent_season": ent_season,
            "ref_season": ref_season,
            "wh_season": wh_season


        }
    )

    return config
