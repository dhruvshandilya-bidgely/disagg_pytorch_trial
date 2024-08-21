
"""
Author - Nisha Agarwal
Date - 9th Feb 20
laundry detection config file
"""

import numpy as np

# import functions from within the project

from python3.config.pilot_constants import PilotConstants

from python3.itemization.aer.functions.get_config import get_hybrid_config


def get_detection_config(pilot_level_config, pilot, samples_per_hour=1):

    """
    Initialize config required for laundry detection module

    Parameters:
        pilot_level_config       (dict)           : Dict containing all hybrid v2 pilot config
        pilot                    (int)            : pilot id
    Returns:
        detection_config         (dict)           : Dict containing all laundry detection related parameters
    """

    detection_config = dict()


    max_monthly_cons = 2000
    min_monthly_cons = 150

    if pilot in PilotConstants.EU_PILOTS:
        max_monthly_cons = 500
        min_monthly_cons = 60

    if pilot in PilotConstants.AUSTRALIA_PILOTS:
        max_monthly_cons = 500
        min_monthly_cons = 100


    detection_config.update({

        "flat_ids": [2, 3, 11, 12],
        "house_ids": [1, 8, 9, 6, 7, 10],
        "flat_ao_lim": 700,
        "house_ao_lim": 6000,
        "cons_limit": 1500,

        "not_known_ao": 1200,
        "not_known_cons": 700,
        "not_known_delta": 400,

        "house_cap": 600,

        "flat_high_occ_delta": 450,
        "flat_high_occ_cons": 500,

        "flat_low_occ_delta": 500,

        "flat_low_occ_cons": 1500,
        "flat_low_occ_cap": 10000,

        "occ_limit": 3,

        "perc_cap_for_max_cons": 70,
        'max_monthly_cons': max_monthly_cons,
        'min_monthly_cons': min_monthly_cons,

        'max_ld_cons': 1000,
        'min_ld_cons': 150,
        'min_ld_len': 1,
        'max_ld_len': 2,
        'active_usage_hours': np.arange(7 * samples_per_hour, 22 * samples_per_hour + 1),
        'min_ld_box_in_a_wind': 3,
        'min_days_frac_for_ld_det': 0.6,

        'pilots_to_check_ao_cons': [5044],

        'cov_thres_for_flat_users': 70,
        'act_prof_thres_for_flat_users': 0.08,
        'cov_thres_for_ind_home_users': 95,
        'act_prof_thres_for_ind_home_users': 0.05,
        'cov_thres_for_lower_user_count': 55,
        'cov_thres_for_higher_user_count': 45,
        'cov_thres': [55, 95],
        'act_prof_thres': 0.05,
        'cov_thres_for_user_count': 85,
        'user_count_thres': 3,
        'lower_cov_thres_for_user_count': 80,
        'energy_thres_for_high_user_count': 250,

    })

    # "flat_ids": list of flat dwelling type ids ,
    # "house_ids": list of independent home dwelling type ids,
    # "flat_ao_lim": threshold to identify that a user has higher chances of being a flat user,
    # "house_ao_lim": threshold to identify that a user has higher chances of being a independent home user,
    # "cons_limit": higher chances of laundry being absent with users with less monthly consumption,
    # "not_known_ao": AO threshold for homes where dwelling type is not known,to determine presence of laundry
    # "not_known_cons": consumption level threshold for homes where dwelling type is not known,to determine presence of laundry
    # "not_known_delta": living load delta threshold for homes where dwelling type is not known,to determine presence of laundry
    # "house_cap": very low cons with independent house dwelling type is not given laundry,
    # "flat_high_occ_delta": living load delta threshold for homes where dwelling type is flat and higher occupants present,
    # "flat_high_occ_cons": consumption level threshold for homes where dwelling type is flat and higher occupants present,
    # "flat_low_occ_delta": living load delta threshold for homes where dwelling type is flat and lesser occupants present,
    # "flat_low_occ_cons": consumption level threshold for homes where dwelling type is flat and lesser occupants present,
    # "flat_low_occ_cap": very high cons with flat dwelling type is given laundry consumption ,
    # "occ_limit": higher chances of giving laundry if occupants count is higher than 3

    hybrid_config = get_hybrid_config(pilot_level_config)

    coverage = hybrid_config.get("coverage")

    # Modifying thresholds based on required coverage numbers

    if coverage > 97:
        detection_config["flat_ao_lim"] = 300
        detection_config["house_ao_lim"] = 4500
        detection_config["cons_limit"] = 1000
        detection_config["house_cap"] = 400
        detection_config["flat_low_occ_delta"] = 300
        detection_config["flat_low_occ_cons"] = 1000
        detection_config["not_known_delta"] = 300
        detection_config["not_known_cons"] = 500

    elif coverage > 93:
        detection_config["flat_ao_lim"] = 500
        detection_config["house_ao_lim"] = 5000
        detection_config["cons_limit"] = 1000
        detection_config["house_cap"] = 500
        detection_config["flat_low_occ_delta"] = 400
        detection_config["flat_low_occ_cons"] = 1200
        detection_config["not_known_delta"] = 300
        detection_config["not_known_cons"] = 600

    elif coverage < 85:
        detection_config["flat_ao_lim"] = min(3000, max(700, 3400 - 32 * coverage))
        detection_config["cons_limit"] = min(2500, max(1500, 3600 - 25 * coverage))

        detection_config["not_known_ao"] = min(2500, max(1200, 3900 - 20 * coverage))
        detection_config["not_known_cons"] = min(1500, max(700, 1800 - 12 * coverage))
        detection_config["not_known_delta"] = min(1000, max(400, 1200 - 10 * coverage))

        detection_config["flat_low_occ_cons"] = min(1000, max(500, 1200 - 8 * coverage))
        detection_config["flat_low_occ_delta"] = min(1000, max(500, 1200 - 8 * coverage))

    # Modifying thresholds based on users with geographies that either have lesser laundry amplitude
    # or low consumption level

    factor = 0.7

    if pilot in PilotConstants.INDIAN_PILOTS:
        factor = 0.6

    if hybrid_config.get('geography') == 'eu' or (pilot in PilotConstants.AUSTRALIA_PILOTS):
        detection_config["flat_ao_lim"] = detection_config["flat_ao_lim"] * factor
        detection_config["cons_limit"] = detection_config["cons_limit"] * factor

        detection_config["not_known_ao"] = detection_config["not_known_ao"] * factor
        detection_config["not_known_cons"] = detection_config["not_known_cons"] * factor
        detection_config["not_known_delta"] = detection_config["not_known_delta"] * factor

        detection_config["flat_low_occ_cons"] = detection_config["flat_low_occ_cons"] * factor
        detection_config["flat_low_occ_delta"] = detection_config["flat_low_occ_delta"] * factor

        detection_config["flat_high_occ_cons"] = detection_config["flat_high_occ_cons"] * factor
        detection_config["flat_high_occ_delta"] = detection_config["flat_high_occ_delta"] * factor

    return detection_config
