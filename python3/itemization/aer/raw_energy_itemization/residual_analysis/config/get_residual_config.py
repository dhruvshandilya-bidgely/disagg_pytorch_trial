
"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Initialize inference engine config dictionary
"""

# Import python packages

import numpy as np


def get_residual_config(samples_per_hour=1, occ_count=1):

    """
    Config file for inference calculation

    Parameters:
        samples_per_hour          (int)           : samples per hour

    Returns:
        res_config                (dict)          : Dict containing all appliance config values
    """

    res_config = dict()

    # constants used for WH addition from hybrid v2 module

    wh_addition_dict = dict({
        "twh_pilots": [20012,  5081, 10019, 5030],
        "all_year_swh_pilots": [5069],
        "high_wh_pilots": [10017],
        "mid_wh_pilots": [10009, 10010, 10011, 10012, 10013, 10014, 10015, 10041, 10046, 10048],
        "low_wh_pilots": [10008, 20006, 5045, 5037, 10035, 10030, 10029],
        "wh_days_thres": 200,
        "winter_months": [1, 2, 3, 11, 12],
        "high_wh_lap_hours": np.arange(3*samples_per_hour, 6*samples_per_hour),
        "low_wh_lap_hours": np.arange(2 * samples_per_hour, 6 * samples_per_hour + 1),
        "swh_hours": np.arange(5 * samples_per_hour, 12 * samples_per_hour + 1),
        "swh_month_thres": 50,
        "wh_month_thres": 200,
        'min_days_required': 60,
        'thin_pulse_min_cons': 250,
        'thin_pulse_max_cons': 800,
        'mid_cov_wh_pilots_min_days_frac': 0.33,
        'low_cov_wh_pilots_min_days_frac': 0.7,
        'min_swh_amp': 300,
        'max_swh_amp': 3000,
        'min_days_frac': 0.33
    })

    res_config.update({
        "wh_addition_dict": wh_addition_dict
    })

    # "wh_addition_pilots": pilots with high wh coverage
    # "swh_pilots": pilots with swh
    # "twh_pilots": pilots with twh
    # "all_year_swh_pilots":  pilots with all year swh(flow whs)
    # "high_wh_pilots": pilots with high wh coverage
    # "mid_wh_pilots":pilots with medium wh coverage
    # "low_wh_pilots": pilots with low wh coverage
    # "wh_days_thres": days limit for disagg wh de
    # "winter_months": moths with swh
    # "high_wh_lap_hours": low activity period hours
    # "low_wh_lap_hours": low activity period hours
    # "swh_hours": swh usage hours

    # constants used for seasonal signature addition in hybrid v2 module

    hvac_dict = dict({
        "season_len_thres": 30,
        'hvac_thres': (1200 * samples_per_hour > 1) + (800 * samples_per_hour == 1),
        'low_hvac_pilots': [20012],
        'max_heat_thres': 600,
        'min_heat_thres': 400,
        'swh_pilots_multiplier': 0.4,
        'low_hvac_multiplier': 2,
        'heat_detection_thres': (1500 * (samples_per_hour > 1)) + (1000 * (samples_per_hour == 1)),
        'swh_multiplier': 0.4,
        'wh_multiplier': 0.7,
        'zero_heat_hours': np.arange(5*samples_per_hour, 22*samples_per_hour),
        'hvac_min_cons': 600,
        'hvac_max_cons': 5000,
        'swh_pilot_hvac_min_cons': 300,
        'temp_thres': 50,
        'heat_temp_thres': 80,
        'wh_min_thres': 500,
        'swh_min_thres': 300,
        'winter_months': [11, 12, 1],
        'swh_monthly_thres': 4000,
        'swh_len_thres': 3,
        'wh_len_thres': 5,
        'heat_len_thres': 3,
        'cool_len_thres': 4,
        'hvac_min_len': 2,
        'max_hvac_amp': 8000
    })

    res_config.update({
        "hvac_dict": hvac_dict
    })

    # constants used for box type signature addition in hybrid v2 module

    box_detection_config = dict({

        'min_box_amp': 400/samples_per_hour,
        'max_box_amp': 20000/samples_per_hour,
        'min_box_len': max(1, 0.5 * samples_per_hour),
        'min_box_len_for_wh_boxes': 4 * samples_per_hour,
        'min_len_for_using_disagg_residual': 180,
        'max_box_len': [4*samples_per_hour, 13*samples_per_hour],
        'min_frac_cons_points_required': 0.95,
        'high_amp_box_thres': 3500,
        'min_cons_for_twh_box_check': 3000

    })

    res_config.update({
        "box_detection_config": box_detection_config
    })

    # constants used for box type signature allocation to laundry/cooking/entertainment category

    occ_based_ld_scaling_factor = [1, 1, 1, 1.1, 1.2, 1.3, 1.4, 1.5][np.digitize(occ_count, [1, 2, 3, 4, 5, 6, 7])]

    stat_app_box_config = dict({

        'max_ld_boxes': 15,
        'max_wh_boxes': 40,
        'max_cook_boxes': 25,
        'max_box_amp': 9000,
        'max_box_len': 4,
        'diff_of_box_count_for_office_goer_user': 5,
        'diff_of_box_count_for_dishwasher': 8,
        'diff_of_box_count_for_drier': 3,
        'cons_bucket': [0.4, 0.7, 1.5, 2],
        'ld_scaling_factor': [0.3, 0.5, 1, 1.2, 1.5],
        'cook_scaling_factor': [0.3, 0.5, 1, 1.3, 1.7],
        'min_box_len': 0.5,
        'act_curve_thres': 0.3,
        'non_wh_hours': np.arange(13 * samples_per_hour, 23 * samples_per_hour + 1),
        'cooking_hours': np.append(np.arange(8 * samples_per_hour, 9 * samples_per_hour + 1),
                                   np.arange(12 * samples_per_hour, 14 * samples_per_hour + 1)),

        'wh_hours': np.arange(6 * samples_per_hour, 9 * samples_per_hour + 1),
        'cons_tou_thres' :0.4,
        'days_thres_for_swh': 200,

        'min_cap_wh': 800 / samples_per_hour,
        'max_cap_wh': 25000 / samples_per_hour,
        'max_len_wh': 4 * samples_per_hour,
        'min_len_wh': 0.25 * samples_per_hour,
        'max_wh_boxes_for_zero_disagg': 55,

        'min_ld_amp' : 200 / samples_per_hour,
        'ld_amp_buffer' : 500 / samples_per_hour,
        'ld_amp_cap' : 2000,
        'occ_based_ld_scaling_factor': occ_based_ld_scaling_factor,

        'occ_based_cook_scaling_factor': [1, 1, 1, 1.1, 1.2, 1.3, 1.4, 1.5][np.digitize(occ_count, [1, 2, 3, 4, 5, 6, 7])],
        'min_cook_amp': 150 / samples_per_hour,
        'cook_amp_buffer': 500 / samples_per_hour,
        'cook_amp_cap': 3500 / samples_per_hour,

    })

    res_config.update({
        "stat_app_box_config": stat_app_box_config
    })

    # constants used for timed appliance type signature addition in hybrid v2 module

    timed_app_det_config = dict({

        'min_days_required_for_pp_detection': 15,
        'max_pp_amp': 4000,
        'min_pp_amp': 300,
        'min_timed_sig_len': 50,
        'len_to_check_on_min_timed_sig_len': 120,
        'max_pp_len': 13,
        'min_pp_len': 1.5,

        'min_twh_ts_level_cons_limit': 1000,
        'twh_ts_level_cons_buffer': 1000,
        'min_twh_seq_len_for_extension': 30,
        'min_twh_seq_frac_days_for_extension': 0.15,

        'perc_cap': 95,
        'min_twh_dur': max(samples_per_hour*1.5, 3),
        'min_twh_days_seq': 60,


        'min_twh_seq_len_for_disagg_extension': 20,
        'min_twh_seq_frac_days_for_disagg_extension': 0.05,

        'twh_conf_level': [0.3, 0.6],
        'min_timed_sig_days': 60,
        'consequetive_days_thres': [40, 60],
        'min_dur_for_inc_run_signature': min(5, max(samples_per_hour*1.5, 3)),

        'max_non_timed_days': 3,

        'min_pp_ts_level_cons_limit': 300,
        'pp_ts_level_cons_buffer': 500,

        'min_pp_seq_len_for_extension': 40,
        'min_pp_seq_frac_days_for_extension': 0.2,
        'min_timed_seq': 50,
        'min_days_for_pp_cons_to_be_present': 150,
        'conf_thres_for_extension': 0.75,
        'hours_thres_for_extension': 18*samples_per_hour,
        'min_days_for_winter_pp_cons': 40,

        'min_days_frac_to_remove_fp_cases': 0.33,
        'lower_perc_cap': 5,
        'upper_perc_cap': 80,
        'upper_perc_cap_wh': 70,
        'min_twh_cons': 400,
        'min_pp_cons': 400,
        'min_cap_for_lower_perc': 2000 / samples_per_hour,

    })

    res_config.update({
        "timed_app_det_config": timed_app_det_config
    })

    ent_cons = dict({
        'baseload_cons_max_cap' : [800 / samples_per_hour, 2500 / samples_per_hour],
        'baseload_cons_lower_cap' : [200 / samples_per_hour, 350 / samples_per_hour],
        'min_cap_for_baseload_cons' :  100,
        'baseload_cons_scaling_factor' :  0.75,
        'activity_curve_thres' :  0.15,
    })

    res_config.update({
        "ent_cons": ent_cons
    })

    return res_config
