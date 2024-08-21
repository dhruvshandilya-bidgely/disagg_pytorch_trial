
"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Initialize inference engine config dictionary
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.pilot_constants import PilotConstants

from python3.itemization.aer.functions.itemization_utils import get_index_array


def get_step3_app_estimation_config():

    """
    Config file for step3(statistical PP/EV/WH) calculation

    Parameters:

    Returns:
        config                (dict)          : Dict containing all appliance config values
    """

    config = dict({
        'perc_thres': 75,
        'min_cons_for_step3_app': 500,
        'min_scal_fac': 1.02,
        'max_scal_fac': 1.5,
        'frac_thres': 0.03,
        'ev_range': np.array([10, 170, 700]) * 1000,
        'pp_range': np.array([20, 100, 600]) * 1000,
        'wh_range': np.array([10, 100, 700]) * 1000,

    })

    return config


def get_inf_config(samples_per_hour=1):

    """
    Config file for inference calculation

    Parameters:
        samples_per_hour          (int)           : samples per hour

    Returns:
        inf_config                (dict)          : Dict containing all appliance config values
    """

    inf_config = dict()

    # constants for ao range calculations

    ao_dict = dict({
        "bc_weightage": np.array([0.005, 0.02, 0.95, 0.02, 0.005]),
        "vacation_count_limit": 15
    })

    inf_config.update({
        "ao": ao_dict
    })

    # constants for ref range calculations

    ref_dict = dict({

        "60_min_mul": 1.18,
        "30_min_mul": 1.1,
        "15_min_mul": 1.05,
        "factor_buc": [1, 1, 1, 1.1, 1.1, 1.1, 1.2, 1.2, 1.2],
        "cons_buc": [400, 700, 1500, 2000, 3000, 4000, 6000],
        'max_day_level_limit_for_scaling_ref': 1500,
        'scaling_factor_for_japan_users': 1.05
    })

    inf_config.update({
        "ref": ref_dict
    })

    # constants for ent range calculations

    ent_dict = dict({
        "non_television_cons_factor": 0.3,
        "ent_amp": [200, 200, 150],
        'zero_ent_days_frac': 0.05,
        'min_disagg_frac': 0.7,
        'zero_ent_hours': np.arange(2 * samples_per_hour, 5 * samples_per_hour + 1),
        'min_ent': 20,
        'max_ent': 90,
        'max_range_multiplier': 2

    })

    inf_config.update({
        "ent": ent_dict
    })

    # constants for pp range calculations

    pp_dict = dict({
        "flat_ids": [2, 3, 10],
        'min_pp_len_for_removing_twh': 2.75,
        'min_pp_conf_for_removing_twh': 0.8,
        'min_pp_amp_for_removing_twh': 1500,
        'wh_buffer_hours': 1500,
        'min_pp_amp': 300,
        'min_pp_days_required': 7,
        'min_days_to_check_seasonality': 240,
        'perc_cap_for_app_pot': 90,
        'conf_score_offset_for_overestimated_points': 0.3,
        'max_gap_in_pp_days': 3,
        'max_allowed_cons_in_pp': 10000,
        'min_thres_on_pp_days_required' : 5,
        'max_thres_on_pp_days_required' : 50,
        'days_frac_for_pp_days_required' : 0.1,

    })

    inf_config.update({
        "pp": pp_dict
    })

    # constants for cooling range calculations

    cool_dict = dict({
        "disagg_conf_thres": 0.65,
        'conf_thres': 0.65,
        'min_disagg_frac_required': 0.5,
        'heavy_hvac_pilot': [10017],
        'hvac_days_frac_thres': [0.7, 0.85, 0.95],
        'min_baseload_perc_val': [10, 5, 0],
        'min_baseload_perc': 20


    })

    inf_config.update({
        "cool": cool_dict
    })

    # constants for heating range calculations

    heat_dict = dict({
        "disagg_conf_thres": 0.65,
        'conf_thres': 0.65,
        'min_disagg_frac_required': 0.5,
        'heavy_hvac_pilot': [10017],
        'min_baseload_perc': 15,
        'cool_days_frac_thres': [0.55, 0.9],
        'min_baseload_perc_val': [5, 0]

    })

    inf_config.update({
        "heat": heat_dict
    })

    # constants for ev range calculations

    ev_dict = dict({
        "flat_ids": [2, 3, 10],
        "allowed_var": 1.30,
        'default_type': 3,
        'ev_box_threshold': 21,
        'eu_pilots': [20012, 20013, 20015],
        'eu_ev_box_threshold': 14,
        'recent_ev_days_thres': 90,
        'min_ev_l1_box_thres': 14,
        'min_ev_l2_box_thres': 10,
        'mtd_ev_box_thres': 3,
        'inc_ev_box_thres': 10,
        'disagg_ev_box_thres': 5,
        'recent_ev_box_thres': 10,
        'l2_box_thres': 3000,
        'max_l2_box_len': 6,
        'tou_buffer_hours': 3,
        'max_box_len_factor': 0.4,
        'min_box_len_factor': 2,
        'max_cons_factor': 0.8,
        'min_cons_factor': 1.5,
        'mtd_max_box_len_factor': 0.25,
        'mtd_min_box_len_factor': 4,
        'bc_min_box_count': 3,
        'user_min_box_count': 5,
        'l1_box_thres': 2500,
        'l1_len_factor': 0.8,
        'mtd_l1_len_factor': 0.5,
        'non_ev_l1_hours': np.arange(7 * samples_per_hour, 19 * samples_per_hour + 1),
        'max_l1_missing_days': 120,
        'max_ev_amp': 170000,
        'mtd_l1_min_len': [5 * samples_per_hour, 5 * samples_per_hour, 5 * samples_per_hour],
        'mtd_l1_max_len': [15 * samples_per_hour, 15 * samples_per_hour, 15 * samples_per_hour],
        'mtd_l2_min_len': [2.5 * samples_per_hour, 1.25 * samples_per_hour, 1.25 * samples_per_hour],
        'mtd_l2_max_len': [6 * samples_per_hour, 6 * samples_per_hour, 6 * samples_per_hour],
        'continous_ev_days_thres': 20,
        'timed_behaviour_thres': 0.95,
        'ev_freq_thres': 13.5,
        'mtd_ev_freq_thres': 16,
        'seasons_comparison_thres': 3,
        'extreme_seasons_comparison_thres': 5,
        'eu_ev_freq_thres': 25,
        'high_hvac_pilots': [10017],
        'high_hvac_pilots_amp_thres': 8000,
        'high_ev_usage_hours': np.arange(0, 24 * samples_per_hour),
        'disagg_l2_min_len': [2 * samples_per_hour, 2 * samples_per_hour, 2 * samples_per_hour],
        'disagg_l2_max_len': [4 * samples_per_hour, 4 * samples_per_hour, 4 * samples_per_hour],
        'l2_max_cap' : [10000 / samples_per_hour, 10000 / samples_per_hour, 20000 / samples_per_hour],
        'l2_min_cap': [3500 / samples_per_hour, 3000 / samples_per_hour, 6000 / samples_per_hour],
        'l2_min_len': [2.5 * samples_per_hour, 1.25 * samples_per_hour, 1.25 * samples_per_hour],
        'l2_max_len': [6 * samples_per_hour, 6 * samples_per_hour, 5 * samples_per_hour],
        'eu_l2_max_len': [9 * samples_per_hour, 9 * samples_per_hour, 9 * samples_per_hour],
        'l1_max_cap': [2800 / samples_per_hour, 3800 / samples_per_hour, 2800 / samples_per_hour],
        'l1_min_cap': [800 / samples_per_hour, 500 / samples_per_hour, 700 / samples_per_hour],
        'l1_min_len': [4 * samples_per_hour, 3.5 * samples_per_hour, 8 * samples_per_hour],
        'l1_max_len': [18 * samples_per_hour, 14 * samples_per_hour, 18 * samples_per_hour],
        'amp_multiplier': 1.1,
        'hist_hsm_min_days': 30,
        'inc_hsm_min_days': 50,
        'disagg_l1_non_ev_hours': np.arange(3 * samples_per_hour, 19 * samples_per_hour + 1),
        'disagg_l2_non_ev_hours': np.arange(6 * samples_per_hour, 19 * samples_per_hour + 1),
        'l2_non_ev_hours': np.arange(6 * samples_per_hour, 22.5 * samples_per_hour + 1),
        'l1_non_ev_hours': np.arange(max(1, 0.5 * samples_per_hour), 22 * samples_per_hour + 1),

        'days_frac_for_recent_days' : 0.75,
        'max_days' : 390,

        'amp_buffer_for_eu_pilots' : 2000 / samples_per_hour,
        'min_amp_l2_charging_boxes' : 3000 / samples_per_hour,
        'min_amp_l2_charging_boxes_for_disagg_user' : 3200 / samples_per_hour,
        'buffer_amp_from_disagg_cons' : 2000 / samples_per_hour,
        'max_buffer_amp' : 300 / samples_per_hour,

    })

    # "flat_ids": flat ids
    # 'default_type': default value of ev charger type
    # 'ev_box_threshold': default min ev box count
    # 'eu_ev_box_threshold': default min ev box count for eu pilots
    # 'recent_ev_days_thres': number of days used to detect recent ev
    # 'min_ev_l1_box_thres': min ev box count for l1 detection
    # 'min_ev_l2_box_thres': min ev box count for l2 detection
    # 'mtd_ev_box_thres': min ev box count in mtd mode
    # 'inc_ev_box_thres': min ev box count in incremental mode
    # 'disagg_ev_box_thres': min ev box count when ev disagg is non-zero
    # 'recent_ev_box_thres': min ev box count for recent ev detection
    # 'l2_box_thres': l2 charger amplitude threshold
    # 'max_l2_box_len': max l2 charger box length
    # 'tou_buffer_hours': buffer hours used to add leftover ev boxes before and after ev usage hours
    # 'mtd_max_box_len_factor': multiplier used for length calculation while adding leftover ev l2 boxxes
    # 'mtd_min_box_len_factor': multiplier used for length calculation while adding leftover ev l2 boxxes
    # 'bc_min_box_count': min box count while adding ev in a billing cycle
    # 'user_min_box_count': min box count while adding ev for user
    # 'l1_box_thres': ev l1 charger amplitude threshoold
    # 'l1_len_factor': multiplier used for length calculation while adding leftover ev l1 boxxes
    # 'mtd_l1_len_factor': multiplier used for length calculation while adding leftover ev l1 boxxes in mtd
    # 'non_ev_l1_hours': hours which are neglected in backup ev l1 module
    # 'max_l1_missing_days': maximum length of chunk to run ev l1 backup module
    # 'max_ev_amp': max capacity of ev charger
    # 'mtd_l1_min_len': min l1 charger length for mtd mode
    # 'mtd_l1_max_len': max l1 charger length for mtd mode
    # 'mtd_l2_min_len': min l2 charger length for mtd mode
    # 'mtd_l2_max_len': max l2 charger length for mtd mode
    # 'continous_ev_days_thres': length of sequence used to check continous ev output,
    # 'ev_freq_thres': ev frequency threshold to exclude sparse ev cases
    # 'mtd_ev_freq_thres': ev frequency threshold in mtd mode
    # 'seasons_comparison_thres': factor used to eliminated season ev detected in hybrid
    # 'eu_ev_freq_thres': ev frequency threshold for eu pilots, since they usually have low frequency
    # 'high_hvac_pilots': pilots with heavy hvac to avoid ev l1 fp cases due to hvac
    # 'high_ev_usage_hours': high ev usage hours
    # 'disagg_l2_min_len':  min l2 charger length when disagg is non-zero
    # 'disagg_l2_max_len': max l2 charger length when disagg is non-zero
    # 'l2_max_cap':  max l2 charger capacity
    # 'l2_min_cap':  min l2 charger capacity
    # 'l2_min_len':  min l2 charger length
    # 'l2_max_len':  max l2 charger length
    # 'eu_l2_max_len':  max l2 charger length for eu pilots, since they have long ev strikes
    # 'l1_max_cap':  max l1 charger capacity
    # 'l1_min_cap':  min l1 charger capacity
    # 'l1_min_len':  min l1 charger length
    # 'l1_max_len':  max l1 charger length
    # 'hist_hsm_min_days': min days required to write hsm in historical mode
    # 'inc_hsm_min_days': min days required to write hsm in historical mode
    # 'disagg_l1_non_ev_hours': low ev l2 usage hours when disagg is non-zero
    # 'disagg_l2_non_ev_hours': low ev l1 usage hours when disagg is non-zero
    # 'l2_non_ev_hours': low ev l2 usage hours
    # 'l1_non_ev_hours': low ev l1 usage hours

    inf_config.update({
        "ev": ev_dict
    })

    # constants for wh range calculations

    wh_dict = dict({

        "heatpump_factor": 0.4,
        "wh_max_amp": 15000,
        "wh_min_amp": 1500,
        "wh_max_len": 4,
        "wh_min_len": 1,
        "wh_disagg_min_len": 0.5,
        "wh_disagg_max_len": 3,
        "act_curve_thres": 0.45,
        "japan_pilots": [10039, 5055, 30001],
        "non_swh_months": [4, 5, 6, 7, 8],
        "non_electric_wh_types": ["SOLAR", "PROPANE", "GAS", "Gas", "SOLID_FUEL", "SOLID_FEUL", "OIL", "Oil", "WOOD", "Wood"],
        "other_type_list": ['OTHERS', 'Others', 'Other', 'OTHER', 'others', 'COMBINED'],
        "max_wh_boxes": 40,
        "max_wh_boxes_for_hld_change": 55,
        "max_wh_amp": 6000,
        "all_year_wh": [5069],
        'swh_amp_thres': 500,
        'wh_amp_thres': 1000,
        'max_seasonal_cons': 10000,
        'flow_max_cons': 10000,
        'wh_max_cons': 5000,
        'flow_thres': 6000,
        'pot_thres': 0.5,
        'wh_extention_amp_thres': 0.9,
        'swh_max_amp': 4000,
        'swh_min_amp': 400,
        'non_wh_hours': np.arange(2 * samples_per_hour, 5 * samples_per_hour + 1),
        'non_swh_hours': np.arange(14 * samples_per_hour, 24 * samples_per_hour),
        'twh_min_cons': 500,
        'twh_min_cons_perc': 60,
        'twh_max_cons_perc': 90,
        "wh_addition_pilots": [10017, 5060, 5044, 5082, 5078, 5077, 5076, 5069],
        'swh_start_hour': 5 * samples_per_hour + 1,
        'swh_end_hour': 20 * samples_per_hour,
        'wh_end_hour': 12 * samples_per_hour,
        'wh_pot_buffer_days': 20,
        'twh_amp_buffer': 1000,
        'twh_max_thres': 7000,
        'twh_min_thres_japan': 1000,
        'twh_min_thres': 1500,
        'twh_dur_thres': 6,
        'twh_dur_thres_japan': 8,
        'tankless_wh_max_cons': 8000,
        'wh_max_ts_cons': 6000,
        'less_days_wh_max_cons': 4000,
        'tankless_min_ts_cons': 1500,
        'min_days_for_lower_ts_level_limit': 150,
        'app_delta_thres': [-30, -20],
        'max_cons_thres': [70000, 40000],
        'max_delta_calc_window_thres': 30,
        'disagg_upper_perc_limit': 99,
        'upper_perc_limit': 98,
        'wh_min_box_req': min(10, 5 * samples_per_hour),
        'twh_min_box_req': min(15, 5 * samples_per_hour),
        'wh_feeble_cons_thres': 5,
        'twh_feeble_cons_thres': 15,
        'swh_min_box_req': 5 * samples_per_hour,
        'tankless_wh_min_box_req': 5 * samples_per_hour,
        'min_box_req': 2.5 * samples_per_hour,
        'feeble_cons_thres': 3,

        'wh_extension_in_summers_box_count_thres': 0.33,
        'min_swh_usage_hour': 5.5,
        'seasonality_thres_for_adjustment': 1.3,
        'seasonality_thres_for_deletion': 1.75,

        'box_add_min_wh_amp': 800,
        'box_add_min_wh_len': 0.25,
        'box_add_min_tankless_wh_amp': 800,
        'box_add_max_wh_amp': 10000,
        'box_add_max_wh_amp_inc': 300,
        'box_add_min_wh_amp_dec': 500,
        'box_add_min_wh_for_nonzero_disagg': 900,

        'disagg_cons_thres': 3000,
        'max_cap_on_minimum_wh_days_required_in_month': 10,
        'min_cap_on_minimum_wh_days_required_in_month': 3,
        'minimum_wh_days_required_in_month': 5,

        'perc_cap_for_app_pot': 90,
        'min_days_required_for_hsm_posting': 15,
        'min_days_to_check_seasonality': 200,

        'item_cons_thres_for_low_cons_wh_blocking': 6000,
        'disagg_cons_thres_for_low_cons_wh_blocking': 2000,

        'thin_pulse_amp_max_thres': 4000,
        'thin_pulse_amp_min_thres': 300,
        'thin_pulse_amp_max_ts_cons': 800,
        'thin_pulse_amp_buffer': 400,
        'thin_pulse_max_amp_factor': 1.2,

        'wh_cons_per_cap': 75,
        'min_wh_amp_for_flow_wh': 6000,

        'min_swh_monthly_cons': 4000,
        'max_jump_allowed_based_on_room_count': [1.3, 1.3, 1.3, 1.3, 1.3, 1.6, 1.8, 2, 2.1, 2.2, 2.5],
        'room_count_buckets': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

        'max_thin_pulse_count': 6,

        'max_freq': 3,
        'non_wh_residual_hours': np.arange(1 * samples_per_hour, 5 * samples_per_hour + 1),
        'additional_night_hours':  np.arange(0, 2 * samples_per_hour + 1),

        'swh_pot_days_buffer': 20,
        'non_swh_wh_hours': get_index_array(22 * samples_per_hour, 5 * samples_per_hour + 1, 24 * samples_per_hour),

        'default_wh_amp_for_box_addition': 1500,
        'allowed_deviation': 10,

        'max_thin_pulse_in_day': 6


    })

    # "heatpump_factor": multiplier for whs with heatpump type
    # "wh_max_amp": max amp for wh box addition
    # "wh_min_amp": min amp for wh box addition
    # "wh_max_len": max length for wh box addition
    # "wh_min_len": min length for wh box addition
    # "wh_disagg_min_len": min length for wh box addition when disagg is non-zero
    # "wh_disagg_max_len": max length for wh box addition when disagg is non-zero
    # "japan_pilots": japan pilot id list
    # "swh_pilots": swh pilot list
    # "non_swh_months": months for which swh is 0
    # "non_electric_wh_types": list of whs type for which electric component is 0

    inf_config.update({
        "wh": wh_dict
    })

    # constants for li range calculations

    li_dict = dict({
        'min_li_cons': 5000,
        'max_scaling_factor': 1.3,
        'min_scaling_factor': 0.7,
        'scaling_factor_3': 0.5,
        'scaling_factor_2': 0.75,
        'scaling_factor_0': 1.75,
        'room_count_bucket': [3, 4, 5, 6],
        'room_count_factor': [1, 1.05, 1.1, 1.15, 1.2],
        'day_level_cap_for_eff_li': 1200,
        'day_level_cap_for_ineff_li': 200


    })

    inf_config.update({
        "li": li_dict
    })

    # constants for app range calculations after handling of negative residual points

    neg_res_handling_dict = dict({
        'score_val_bucket_for_box_hvac_adjusment': [1, 0.9, 0.6, 0.4, 0.2, 0.15, 0],
        'conf_val_bucket_for_box_hvac_adjusment': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'conf_thres_for_non_ev_users': 0.75,
        'conf_thres_for_ev_users': 0.6,
        'score_val_bucket_for_timed_hvac_adjusment': [1, 0.95, 0.9, 0.7, 0.5, 0.3, 0.15, 0],
        'conf_val_bucket_for_timed_hvac_adjusment': [0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'conf_thres_for_wh': 0.75,
        'night_hours': np.append(np.arange(0 * samples_per_hour, 3 * samples_per_hour + 1).astype(int),
                                 np.arange(22 * samples_per_hour, 24 * samples_per_hour).astype(int)),
        'ev_wh_min_conf': 0.3,
        'min_wh_amp': 4000 / samples_per_hour,
        'wh_conf_offset_for_night_hours': 0.5,
        'wh_conf_offset_for_high_cons_points': 0.3,
        'ev_conf_thres': 0.9,
        'wh_conf_for_high_conf_ev_points': 0.7

    })

    inf_config.update({
        "neg_res_handling": neg_res_handling_dict
    })

    return inf_config


def get_ev_params_for_det(samples, ev_disagg, item_input_object):

    """
    fetching required inputs for timed signature detection

    Parameters:
        samples                   (int)           : samples in an hour
        ev_disagg                 (double)        : total ev disagg
        item_input_object         (dict)          : Dict containing all inputs

    Returns:
        params                    (dict)          : Ev parametere for addition of ev strikes in hybrid
    """

    min_len = [2, 2, 4, 5][np.digitize(samples, [1, 2, 4])]

    low_dur_box_thres = [1, 1, 3, 4][np.digitize(samples, [1, 2, 4])]

    if np.sum(ev_disagg) == 0:
        low_dur_box_thres = [2, 2, 4, 4][np.digitize(samples, [1, 2, 4])]


    if item_input_object.get("config").get("pilot_id") in PilotConstants.EU_PILOTS:
        max_len = samples*12
    else:
        max_len = samples*11


    diff = 0

    if item_input_object.get("config").get("pilot_id") in PilotConstants.EU_PILOTS:
        diff = 1000

    if samples == 1:
        diff = -1300

    initial_amp = (2700-diff)/samples

    ev_amp = -1

    if item_input_object.get("config").get('disagg_mode') in ['incremental', 'mtd'] and \
            item_input_object.get("item_input_params").get('ev_hsm') is not None \
            and item_input_object.get("item_input_params").get('ev_hsm').get('item_amp') is not None:
        ev_amp = item_input_object.get("item_input_params").get('ev_hsm').get('item_amp')

        if isinstance(ev_amp, list):
            ev_amp = ev_amp[0] / samples
        else:
            ev_amp = ev_amp / samples

    if np.any(ev_disagg > 0):
        max_amp_for_disagg_users = np.percentile(ev_disagg[ev_disagg > 0], 95) * 1.3
        min_amp_for_disagg_users = np.percentile(ev_disagg[ev_disagg > 0], 90) * 0.9 - 1000 / samples
    else:
        max_amp_for_disagg_users = 0
        min_amp_for_disagg_users = 0

    min_amp = ev_amp * 0.9 - 2000 / samples
    max_amp = ev_amp * 1.3


    min_len_for_cons = [1, 1, 3, 4][np.digitize(samples, [1, 2, 4])]

    if item_input_object.get("config").get("pilot_id") in PilotConstants.EU_PILOTS:
        max_len_for_cons = samples * 12
    else:
        max_len_for_cons = samples * 8


    params = {
        'min_len': min_len,
        'low_dur_box_thres': low_dur_box_thres,
        'max_len': max_len,
        'initial_amp': initial_amp,
        'ev_amp': ev_amp,
        'max_amp_for_disagg_users': max_amp_for_disagg_users,
        'min_amp_for_disagg_users': min_amp_for_disagg_users,
        'max_amp': max_amp,
        'min_amp': min_amp,
        'ev_amp_cap': 30000,
        'high_freq_days_thres': 0.15,
        'high_cons_ev_thres': 5000/samples,
        'min_len_for_cons': min_len_for_cons,
        'max_len_for_cons': max_len_for_cons

    }

    return params
