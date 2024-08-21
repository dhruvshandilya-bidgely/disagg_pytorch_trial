
"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Initialize inference engine config dictionary
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.itemization.aer.functions.get_config import get_hybrid_config


def get_inf_config2(item_input_object, samples_per_hour=1):

    """
    Config file for inference calculation

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        samples_per_hour          (int)           : samples per hour

    Returns:
        inf_config                (dict)          : Dict containing all appliance config values
    """

    inf_config = dict()

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    cook_idx = hybrid_config.get("app_seq").index('cook')

    scale_cons = hybrid_config.get("scale_app_cons")[cook_idx]
    cons_factor = hybrid_config.get("scale_app_cons_factor")[cook_idx]

    if scale_cons:

        if cons_factor > 3:
            cooking_hours = np.append(np.arange(6 * samples_per_hour, 12 * samples_per_hour + 1),
                                      np.arange(12 * samples_per_hour, 14 * samples_per_hour + 1))
            cooking_hours = np.append(cooking_hours,
                                      np.arange(17 * samples_per_hour, 21 * samples_per_hour + 1))

            cooking_hours = cooking_hours.astype(int)

        elif cons_factor >= 2.5:

            cooking_hours = np.append(np.arange(6 * samples_per_hour, 10 * samples_per_hour + 1),
                                      np.arange(12 * samples_per_hour, 14 * samples_per_hour + 1))
            cooking_hours = np.append(cooking_hours,
                                      np.arange(17 * samples_per_hour, 20 * samples_per_hour + 1))

            cooking_hours = cooking_hours.astype(int)
        else:

            cooking_hours = np.append(np.arange(6 * samples_per_hour, 9 * samples_per_hour + 1),
                                      np.arange(12 * samples_per_hour, 14 * samples_per_hour + 1))
            cooking_hours = np.append(cooking_hours,
                                      np.arange(18 * samples_per_hour, 20 * samples_per_hour + 1))

            cooking_hours = cooking_hours.astype(int)
    else:

        cooking_hours = np.append(np.arange(6 * samples_per_hour, 9 * samples_per_hour + 1),
                                  np.arange(12 * samples_per_hour, 14 * samples_per_hour + 1))
        cooking_hours = np.append(cooking_hours,
                                  np.arange(18 * samples_per_hour, 20 * samples_per_hour + 1))

        cooking_hours = cooking_hours.astype(int)

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
        "cons_buc": [400, 700, 1500, 2000, 3000, 4000, 6000]
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
        'max_range_multiplier': 2,
        'default_sleep_hours': np.arange(1 * samples_per_hour, 6 * samples_per_hour),
        'max_ent_cons_cap': 100,
        'max_ent_cons_offset': 200,
        'min_ent_cons_cap': 300,
        'min_ent_cons_offset': 500,
        'perc_cap_for_ts_level_cons': 95

    })

    inf_config.update({
        "ent": ent_dict
    })

    # constants for ld range calculations

    cooking_hours = np.arange(7 * samples_per_hour, 22 * samples_per_hour + 1)

    ld_dict = dict({
        "cooking_hours": cooking_hours,
        "app_tou_factor": 0.05,
        "overest_tou_factor": 0.1,
        "ld_amp": [2500, 2500, 3000],
        "ld_max_amp": 3000,
        "ld_min_amp": 500,
        "ld_min_len": 1,
        "ld_max_len": 3,
        "ld_hours": np.arange(11 * samples_per_hour, 22 * samples_per_hour + 1),
        'zero_ld_days_frac': 0.05,
        'min_disagg_frac': 0.7,
        'zero_ld_hours': np.arange(int(3 * samples_per_hour), 6 * samples_per_hour + 1),
        'min_ld': 20,
        'max_ld': 90,
        'max_range_multiplier': 2,
    })

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    ld_idx = hybrid_config.get("app_seq").index('ld')
    apply_max_amp = hybrid_config.get("scale_app_cons")[ld_idx]
    max_amp_conf = hybrid_config.get("scale_app_cons_factor")[ld_idx]

    ld_idx = hybrid_config.get("app_seq").index('ld')

    scale_cons = hybrid_config.get("scale_app_cons")[ld_idx]
    cons_factor = hybrid_config.get("scale_app_cons_factor")[ld_idx]

    if apply_max_amp:
        ld_dict["ld_amp"] = np.array(ld_dict["ld_amp"]) * max_amp_conf
        ld_dict["ld_max_amp"] = np.array(ld_dict["ld_max_amp"]) * max_amp_conf

    if scale_cons:
        if cons_factor > 2:
            ld_dict["ld_max_amp"] = 6000

        elif cons_factor > 1.7:
            ld_dict["ld_max_amp"] = 5000

        elif cons_factor > 1.5:
            ld_dict["ld_max_amp"] = 4000

        elif cons_factor > 1.2:
            ld_dict["ld_max_amp"] = 3000


    inf_config.update({
        "ld": ld_dict
    })

    # constants for cooking range calculations

    cook_dict = dict({
        "cooking_hours": cooking_hours,
        "app_tou_factor": 0.05,
        "overest_tou_factor": 0.3,
        "min_cons_factor": 0.5,
        "cooking_amp": [250, 700, 100],
        "gas_multiplier": 0.1,
        "cook_min_amp": 800,
        "cook_max_amp": 4000,
        "cook_min_len": 0.25,
        "cook_max_len": 2,
        'zero_cook_days_frac': 0.05,
        'min_disagg_frac': 0.8,
        'zero_cook_hours': np.arange(1 * samples_per_hour, 6 * samples_per_hour + 1),
        'min_cook': 20,
        'max_cook': 90,
        'max_range_multiplier': 2,
        'min_res_wh': 800,
        'max_res_wh': 10000,
        'default_wakeup_time': int(6 * samples_per_hour),
        'additional_cook_hours': np.arange(19 * samples_per_hour, 21 * samples_per_hour + 1),
        'sleep_hours_thres': 4 * samples_per_hour
    })

    # constants for WH range calculations

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
        "non_swh_months": [4, 5, 6, 7, 8, 9],
        "non_electric_wh_types": ["SOLAR", "PROPANE", "GAS", "Gas", "SOLID_FUEL", "SOLID_FEUL", "OIL", "Oil", "WOOD", "Wood"],
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
        'non_swh_hours': np.arange(14 * samples_per_hour, 24 * samples_per_hour)

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

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    ld_idx = hybrid_config.get("app_seq").index('cook')
    apply_max_amp = hybrid_config.get("scale_app_cons")[ld_idx]
    max_amp_conf = hybrid_config.get("scale_app_cons_factor")[ld_idx]

    if apply_max_amp:
        cook_dict["cooking_amp"] = np.array(cook_dict["cooking_amp"]) * max_amp_conf
        cook_dict["cook_max_len"] = min(2.5 * samples_per_hour, np.array(cook_dict["cook_max_len"]) * max_amp_conf)

    inf_config.update({
        "cook": cook_dict
    })

    return inf_config
