
"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Initialize config dictionary for performing 100 % itemizations
"""

import numpy as np


def init_final_item_conf(total_monthly_cons=0, actual_total=0):

    """
    Config file for final appliance adjustments

    Returns:
        itemization_config        (dict)          : Dict containing all appliance config values
    """

    itemization_config = dict()

    res_conf = dict({
        "min_res": 10,
        "min_res_in_overest": 10
    })

    itemization_config.update({
        "res_conf": res_conf
    })

    sat_conf = dict({
        "inc_min": 0.93,
        "inc_mid": 0.99,
        "inc_max": 0.995,

        "dec_min": 0.93,
        "dec_mid": 0.99,
        "dec_max": 0.995,

        "cons_min": 0.93,
        "cons_mid": 0.99,
        "cons_max": 0.995,

        "sleep_min": 0.95,
        "sleep_mid": 0.99,
        "sleep_max": 1
    })


    itemization_config.update({
        "sat_conf": sat_conf
    })

    # config required to handle overestimation points

    case_a_conf = dict({
        "twh_app_seq": ['ao', 'ref', 'pp', 'wh', 'ev',  'cooling', 'heating', 'ld', 'cook', 'ent', 'li',],
        "non_twh_app_seq": ['ao', 'ref', 'pp', 'ev', 'wh', 'cooling', 'heating', 'ld', 'cook', 'ent', 'li',]
    })

    itemization_config.update({
        "case_a_conf": case_a_conf
    })

    # config required to handle underestimation points

    case_b_conf = dict({
        "twh_app_seq": ['ao', 'ref', 'pp', 'wh', 'ev', 'cooling', 'heating',  'ld', 'cook', 'ent', 'li',],
        "non_twh_app_seq": ['ao', 'ref', 'pp', 'ev', 'wh', 'cooling', 'heating',  'ld', 'cook', 'ent', 'li',]
        })

    itemization_config.update({
        "case_b_conf": case_b_conf
    })

    # config required to handle overestimation points

    case_c_conf = dict({
        "twh_app_seq": [ 'ld', 'cook', 'ent', 'li', 'heating', 'cooling',  'ev', 'pp', 'wh'],
        "non_twh_app_seq": [ 'ld', 'cook', 'ent', 'li', 'heating', 'cooling', 'wh', 'ev', 'pp']
    })

    itemization_config.update({
        "case_c_conf": case_c_conf
    })

    post_processing_config = dict({

        'max_wh_delta': 75,
        'max_swh_delta': 1000,
        'swh_pilots': [5069],
        'max_mtd_wh_delta': 60,

        'pp_conf_buc_for_single_speed_pp': [0.7, 0.9],
        'perc_buc_for_single_speed_pp': [40, 50, 75],
        'min_pp_cons_for_single_speed_pp': 400,

        'pp_conf_buc_for_multi_speed_pp': [0.7],
        'perc_buc_for_multi_speed_pp': [40, 60],
        'perc_cap_for_multi_amp_pp_flag': [25, 90],
        'min_diff_for_multi_amp_pp_flag': 600,
        'min_days_to_check_min_pp_cons': 250,
        'min_detection_conf': 0.75,
        'max_pp_usage_hours': 18,
        'min_pp_days_required': 90

    })

    itemization_config.update({
        "post_processing_config": post_processing_config
    })

    # config for hsm posting

    hsm_config = dict({
        'hist_hsm_min_days': 30,
        'inc_hsm_min_days': 50,
    })

    itemization_config.update({
        "hsm_config": hsm_config
    })


    limit_perc = 10 - ((total_monthly_cons - 3000) / 170)
    limit_cons = (total_monthly_cons + 100) * 0.1

    limit_perc = min(limit_perc, 60)
    limit_perc = max(limit_perc, 40)

    limit_cons = min(limit_cons, 500)
    limit_cons = max(limit_cons, 5)

    if total_monthly_cons < 400:
        ent_max_limit = max(limit_cons, total_monthly_cons * limit_perc / 100) * 1.1
        cook_max_limit = max(limit_cons, total_monthly_cons * limit_perc / 100) * 0.8
        ld_max_limit = max(limit_cons, total_monthly_cons * limit_perc / 100) * 1.2

    else:

        limit_perc = 10 - ((total_monthly_cons - 3000) / 170)
        limit_cons = (total_monthly_cons + 100) * 0.2

        limit_perc = min(limit_perc, 60)
        limit_perc = max(limit_perc, 40)

        limit_cons = min(limit_cons, 500)
        limit_cons = max(limit_cons, 5)

        ent_max_limit = min(limit_cons, total_monthly_cons * limit_perc / 100) * 1.1
        cook_max_limit = min(limit_cons, total_monthly_cons * limit_perc / 100) * 0.8
        ld_max_limit = min(limit_cons, total_monthly_cons * limit_perc / 100) * 1.2

    # Function to determine min consumption limit based on the bc level monthly cosumption output
    limit_perc = 1 - ((3000 - total_monthly_cons) / 1000)
    limit_cons = (total_monthly_cons - 2000) / 200 + 50

    limit_perc = min(limit_perc, 5)
    limit_perc = max(limit_perc, 0.3)

    limit_cons = min(limit_cons, 60)
    limit_cons = max(limit_cons, 2)

    # min consumption limit for individual appliances

    ent_min_limit = min(limit_cons, total_monthly_cons * limit_perc / 100)
    cook_min_limit = min(limit_cons, total_monthly_cons * limit_perc / 100) * 0.8
    ld_min_limit = min(limit_cons, total_monthly_cons * limit_perc / 100) * 1.2

    limit_cons = (total_monthly_cons - 5000) / 200 + 50
    limit_cons = min(90, limit_cons)
    limit_cons = max(limit_cons, 2)

    limit_cons = min(limit_cons, [8, 8, 15, 20, 25, 30, 40, 50, 70, 80][np.digitize(actual_total, [300, 400, 700, 1000, 1500, 2000, 3000, 4000, 6000])])

    ent_limit = limit_cons
    cook_limit = limit_cons * 0.8
    ld_limit = limit_cons * 1.3

    # config required to maintain bc level min cons of stat app

    min_max_limit_conf = dict({
        'ent_max_limit': ent_max_limit,
        'cook_max_limit': cook_max_limit,
        'ld_max_limit': ld_max_limit,

        'ent_min_limit': ent_min_limit,
        'cook_min_limit': cook_min_limit,
        'ld_min_limit': ld_min_limit,

        'ent_room_count_buc' : [1, 3, 4],
        'ent_room_count_scaling_fac' : [1, 0.8, 1, 1.2],
        'ent_occ_count_buc' : [1, 3, 4],
        'ent_occ_count_scaling_fac' : [1, 0.7, 1, 1.2],

        'ld_room_count_buc': [1, 3, 4],
        'ld_room_count_scaling_fac': [1, 0.8, 1, 1.2],
        'ld_occ_count_buc': [1, 3, 4],
        'ld_occ_count_scaling_fac': [1, 0.7, 1, 1.2],
        'ld_app_count_buc': [0.5, 1.5, 2],
        'ld_app_count_scaling_fac': [0.7, 1, 1.7, 2],

        'cook_room_count_buc': [1, 3, 4],
        'cook_room_count_scaling_fac': [1, 0.8, 1, 1.2],
        'cook_occ_count_buc': [1, 3, 4],
        'cook_occ_count_scaling_fac': [1, 0.7, 1, 1.2],
        'cook_app_count_buc': [0.5, 1.2, 1.75],
        'cook_app_count_scaling_fac': [0.7, 1, 1.5, 2],

        'max_cap_on_hard_limit' : 200,
        'high_cook_cons_bucket' : [75, 100],
        'high_cook_cons_scaling_factor': [1, 1.2, 1.1*1.2],

        'cons_offset_for_eu' : 30,
        'cons_offset_for_ind': 20,
        'cons_offset' : 70,
        'cons_multiplier': 1.5,
        'ent_limit' : ent_limit,
        'cook_limit': cook_limit,
        'ld_limit' : ld_limit,

    })

    itemization_config.update({
        "min_max_limit_conf": min_max_limit_conf
    })

    return itemization_config
