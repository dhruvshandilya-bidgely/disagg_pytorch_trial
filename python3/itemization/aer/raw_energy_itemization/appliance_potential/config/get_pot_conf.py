
"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Initialize potential calculation engine config dictionary
"""

import numpy as np


def get_pot_conf(samples=1):

    """
     Config file for ts level potential calculation

    Parameters:
        samples           (int)           : number of data samples in an hour

    Returns:
        pot_config        (dict)          : Dict containing all appliance config values
    """

    pot_config = dict()

    ao_dict = dict({

    })

    pot_config.update({
        "ao": ao_dict
    })

    ref_dict = dict({
        "season_factor": [1.1, 1.05, 1, 0.95, 0.9],
        "conf_intercept": 0.35
    })

    pot_config.update({
        "ref": ref_dict
    })

    ent_dict = dict({
        "base_potential": 0.7,
        "weekend_day_inc": 0.2,
        'ent_cons_score_for_late_night_hours': 0.8,
        'ent_cons_potential_score_offset': 0.4,
        'activity_curve_thres': [0.25, 0.3],
        'activity_curve_cons_thres': [0, 4000]

    })

    pot_config.update({
        "ent": ent_dict
    })

    pp_dict = dict({

        "min_days_in_run_type": 15,
        "run_type_days_win": 20,
        "yearly_tou_inc": 0.25,
        'pp_confidence_score_offset': 0.1,
        'perenial_load_weight_in_pp_conf': 0.7,
        'seasonal_potential_weight_in_pp_conf': 0.3,
        'disagg_conf': [0.75, 0.7, 0.8],
        'min_hyrbid_conf': [0.8, 0.6, 0.85],
        'wind_size_for_season_wise_score': 15,
        'max_cons_perc': 90,
        'default_disagg_confidence': 0.8,
        'pp_conf_buckets' : [0.6, 0.7, 0.8, 0.9],
        'pp_perenial_score_offset' : 0.3,
        'season_based_score_offset' : 0.2,

    })

    pot_config.update({
        "pp": pp_dict
    })

    twh_dict = dict({

        "min_days_in_run_type": 15,
        "run_type_days_win": 20,
        "yearly_tou_inc": 0.25

    })

    pot_config.update({
        "twh": twh_dict
    })

    li_dict = dict({
        "min_cloud_cover_in_active_hour": 0.9,
        "min_cloud_cover": 0.7
    })

    pot_config.update({
        "li": li_dict
    })

    ld_dict = dict({
        'conf_score_at_non_zero_disagg': 0.7,
        'weekdays_ld_potential_score_buffer': 0.3,
        'weekends_ld_potential_score_buffer': 0.2

    })

    pot_config.update({
        "ld": ld_dict
    })

    ev_dict = dict({

        "valley_score_weight": 0.1,
        "disagg_weight": 0.9,
        "active_hours_inc": 0.1,
        "start_score_arr": [1, 1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.1, -1, -1, 0.7, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1, 1, 1, 1, 1],
        "end_score_arr": [0.8, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.8, 0.7, 0.5, 0.4, 0.4, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
        "length_score_arr": [-5, -5, -5, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.8, 0.6, 0.5, 0.5, 0.5, 0.5],
        "amp_score_arr": [0.5, 0.5, 0.5, 0.5, 0.7, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.8, 0.7,
                          0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        'ev_score_thres': 0.3,

        'ev_l1_tag_threshold': 3000,
        'season_potential_score_bucket': [0.75, 0.7],
        'consistent_usage_score_bucket': [0.85, 0.85],
        'min_conf_for_disagg_users': 0.7,
        'min_ev_conf': 0.3

    })

    pot_config.update({
        "ev": ev_dict
    })

    wh_dict = dict({
        "act_curve_inc": 0.5,
        "start_score_weight": 0.5,
        "consistent_usage_score_weight": 0.5,
        "disagg_weight": 0.5,
        "valley_score_weight": 0.0,
        "seasonal_pot_weight": 0.5,
        "max_conf": 0.9,

        'perc_used_to_get_max_val': 99,
        'conf_score_of_thin_pulse_cons': 0.95,
        'min_wh_conf_score': 0.3,
        'min_seasonal_potential': 0.8,

        'disagg_conf': [0.65, 0.6, 0.7],
        'min_hyrbid_conf': [0.8, 0.7, 0.85],
        'potential_score_offset': [0.2, 0.2, 0.1],
        'twh_conf_offset': 0.3,
        'twh_perenial_score_offset': 0.2,
        'twh_disagg_conf_offset': 0.15,
        'min_frac_for_high_twh_days_count': 0.9,
        'conf_offset_high_twh_days_count': 0.2,
        'min_twh_conf': 0.15

    })

    pot_config.update({
        "wh": wh_dict
    })

    hvac_dict = dict({
        'heating_low_probable_hours': np.arange(5 * samples, 10 * samples + 1),
        'offset_for_low_probable_heating_hours': 0.4,
        'conf_score_weights_for_perenial_usage_user': [0.35, 0.15, 0.15, 0.35],
        'conf_score_weights': [0.7, 0.15, 0.15],
        'pot_score_weights': [0.3, 0.7],
        'thres_for_low_hvac_points_adjusment': 0.3,
        'thres_for_perennial_cons_score_adjusment': 0.4,
        'cooling_score_offset': 0.1,
        'perenial_score_thres': 30

    })

    pot_config.update({
        "hvac": hvac_dict
    })

    cook_dict = dict({

        'cooking_potential_score_offset': 0.2,
        'activity_curve_thres': [0.4, 0.5],
        'activity_curve_cons_thres': [0, 4000],
        'non_cooking_hours': np.arange(1*samples, 5*samples + 1),
        'cooking_pot_score_at_nonzero_disagg_points': 0.6,
        'cooking_cons_scaling_factor': [2.2, 3],
        'cooking_cons_scaling_factor_offset': 0.1

    })

    pot_config.update({
        "cook": cook_dict
    })

    return pot_config
