
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Cooking config file
"""
import copy

import numpy as np

from python3.config.pilot_constants import PilotConstants

from python3.itemization.aer.functions.get_config import get_hybrid_config

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config2 import get_inf_config2


def get_cooking_config(item_input_object, pilot_level_config, samples_per_hour):

    """
    Initialize config required for cooking ts level estimation module

    Parameters:
        item_input_object      (dict)           : dict containing all hybrid inputs
        pilot_level_config     (dict)           : hybrid config for the given pilot user
        samples_per_hour       (int)            : Total number of samples in an hour

    Returns:
        cooking_config         (dict)           : Dict containing all cooking related parameters
    """

    hybrid_config = get_hybrid_config(pilot_level_config)

    cook_idx = hybrid_config.get("app_seq").index('cook')

    scale_cons = hybrid_config.get("scale_app_cons")[cook_idx]
    cons_factor = hybrid_config.get("scale_app_cons_factor")[cook_idx]

    cooking_config = dict()

    # "breakfast_hours": Breakfast hours
    # "lunch_hours":lunch hours
    # "dinner_hours": dinner hours
    # "energy_profile_limit": modifying cooking delta  for low ranges , and high variation in energy profile
    # "delta_limit": modifying cooking delta  for low ranges
    # "delta_multiplier": modifying cooking delta  for low ranges
    # "weekend_present_multiplier": multiplier for users who stays at home on weekends
    # "weekend_absent_multiplier": multiplier for users who stays out on weekends
    # "cooking_absent_multiplier": 0 cooking app count multiplier

    general_config = {
        "breakfast_hours": np.arange(6 * samples_per_hour, 9.5 * samples_per_hour + 1),
        "lunch_hours": np.arange(11.5 * samples_per_hour, 14 * samples_per_hour + 1),
        "dinner_hours": np.arange(18 * samples_per_hour, 21 * samples_per_hour + 1),
        "energy_profile_limit": [400, 700],
        "delta_limit": 200,
        "delta_multiplier": [1.2, 1.4],
        "weekend_present_multiplier": 1.3,
        "weekend_absent_multiplier": 0.5,
        "cooking_absent_multiplier": 0.1,
        'scaling_factor_for_energy_delta': max(1, samples_per_hour / 1.3)
    }

    # "appliance_consumption": appliances capacity for all 3 categories of cooking
    # "delta_arr": estimation of cooking app count (if not given) based on energy  delta
    # "cooking_app_count_arr": estimation of cooking app count (if not given) based on energy  delta
    # "gas_cooking_multiplier": multiplier fo gas cooking
    # "usage_hours": breakfast cooking hours
    # "breakfast_end_time": max breakfast time
    # "occupants_limit": higher cooking for users with > 3 occupants
    # "high_occupants_multiplier": high users multiplier
    # "late_wakeup_consumption": breakfast usage for late wakeup users
    # "breakfast_buffer_hours": breakfast  cooking hours buffer (before leaving for office)

    breakfast_config = {
        "appliance_consumption": [400, 450, 200],
        "delta_arr": [100, 200, 250, 500],
        "cooking_app_count_arr": [[0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 1, 3], [1, 1, 3]],
        "gas_cooking_multiplier": 0.1,
        "usage_hours": 1,
        "breakfast_end_time": 11,
        "occupants_limit": 3,
        "high_occupants_multiplier": 1.15,
        "late_wakeup_consumption": 200,
        "breakfast_buffer_hours": 0.5,
        "60_min_multiplier": 0.8
    }

    # "appliance_consumption": appliances capacity for all 3 categories of cooking
    # "delta_arr": estimation of cooking app count (if not given) based on energy  delta
    # "cooking_app_count_arr": estimation of cooking app count (if not given) based on energy  delta
    # "gas_cooking_multiplier": multiplier fo gas cooking
    # "usage_hours": lunch cooking hours
    # "high_stay_at_home_usage_hours": usage hours for users who stays at  home ,
    # "occupants_limit": higher cooking for users with > 2 occupants
    # "high_stay_at_home_multiplier": multiplier for users who stays at  home ,
    # "absent_lunch_consumption": consumption for users with afternoon band missing
    # "default_lunch_start": default lunch cooking  start time

    lunch_config = {
        "appliance_consumption": [300, 1000, 200],
        "delta_arr": [150, 250, 350, 700, 1100],
        "cooking_app_count_arr": [[0, 0, 1], [0, 0, 2], [1, 0, 3], [1, 0.5, 3], [1, 1, 3], [1.5, 1, 3]],
        "gas_cooking_multiplier": 0.1,
        "usage_hours": 1.5,
        "high_stay_at_home_usage_hours": 1.5,
        "occupants_limit": 2,
        "high_stay_at_home_multiplier": 1.15,
        "absent_lunch_consumption": 200,
        "default_lunch_start": 12,
        "60_min_multiplier": 0.8
    }

    # "appliance_consumption": appliances capacity for all 3 categories of cooking
    # "delta_arr": estimation of cooking app count (if not given) based on energy  delta
    # "cooking_app_count_arr": estimation of cooking app count (if not given) based on energy  delta
    # "gas_cooking_multiplier": multiplier fo gas cooking
    # "usage_hours": dinner cooking hours
    # "high_office_goers_usage_hours": dinner cooking hours for office going people
    # "occupants_limit": higher cooking for users with > 3 occupants
    # "high_occupants_multiplier": higher cooking for users with > 3 occupants
    # "absent_dinner_consumption": consumption for users with evening band missing
    # "default_dinner_start": default dinner cooking start time
    # "default_dinner_end": default dinner cooking end time

    dinner_config = {
        "appliance_consumption": [350, 900, 200],
        "delta_arr": [100, 250, 350, 700, 950, 1200],
        "cooking_app_count_arr": [[0, 0, 1], [0, 0, 2], [1, 0, 2], [1, 0.5, 2], [1, 1, 2], [1, 1, 3], [1.5, 1, 3]],
        "gas_cooking_multiplier": 0.1,
        "usage_hours": 1.5,
        "high_office_goers_limit": 2,
        "high_office_goers_usage_hours": 1.5,
        "occupants_limit": 3,
        "high_occupants_multiplier": 1.15,
        "absent_dinner_consumption": 250,
        "default_dinner_start": 18,
        "default_dinner_end": 21,
        "late_night_hour": 4,
        "60_min_multiplier": 0.8
    }

    # Scaling of box amplitude and box duration of cooking cons based on pilot level cooking averages
    # In pilots where cooking averages are high, higher amount cooking boxes or longer duration of boxes are added

    if scale_cons:
        dinner_config["appliance_consumption"] = np.array(dinner_config["appliance_consumption"]) * cons_factor
        breakfast_config["appliance_consumption"] = np.array(breakfast_config["appliance_consumption"]) * cons_factor
        lunch_config["appliance_consumption"] = np.array(lunch_config["appliance_consumption"]) * cons_factor

        scaling_factor = [1, 1.5, 2, 3, 3.5][np.digitize(cons_factor, [1.5, 2.2, 3, 4])]
        scaling_factor = np.ceil(scaling_factor)

        dinner_config["usage_hours"] = np.array(dinner_config["usage_hours"]) * scaling_factor
        breakfast_config["usage_hours"] = np.array(breakfast_config["usage_hours"]) * scaling_factor
        lunch_config["usage_hours"] = np.array(lunch_config["usage_hours"]) * scaling_factor

    pilot = item_input_object.get("config").get("pilot_id")

    if pilot in PilotConstants.INDIAN_PILOTS:
        dinner_config["usage_hours"] = np.array(dinner_config["usage_hours"]) * 0.5
        breakfast_config["usage_hours"] = np.array(breakfast_config["usage_hours"]) * 0.5
        lunch_config["usage_hours"] = np.array(lunch_config["usage_hours"]) * 0.5

    # updating cooking config based on cooking app profile information

    cooking_app_prof_present_in_survey_input = not item_input_object.get("appliance_profile").get("default_cooking_flag")

    if cooking_app_prof_present_in_survey_input:
        cooking_app_count = copy.deepcopy(item_input_object.get("appliance_profile").get("cooking"))
        cooking_app_type = item_input_object.get("appliance_profile").get("cooking_type")

        if str(np.nan_to_num(item_input_object.get('pilot_level_config').get('cook_config').get('type'))) == 'GAS':
            cooking_app_count[cooking_app_type == 2] = cooking_app_count[cooking_app_type == 2] * 2
            cooking_app_count[cooking_app_type == 0] = item_input_object.get("appliance_profile").get("default_cooking_count")[cooking_app_type == 0]
        else:
            cooking_app_count[cooking_app_type == 0] = cooking_app_count[cooking_app_type == 0] * 0

        max_cook_boxes = np.sum(cooking_app_count) / np.sum(item_input_object.get("appliance_profile").get("default_cooking_count"))

        dinner_config["appliance_consumption"] = np.array(dinner_config["appliance_consumption"]) * max(1, max_cook_boxes / 2)
        breakfast_config["appliance_consumption"] = np.array(breakfast_config["appliance_consumption"]) * max(1, max_cook_boxes / 2)
        lunch_config["appliance_consumption"] = np.array(lunch_config["appliance_consumption"]) * max(1, max_cook_boxes / 2)

        if max_cook_boxes >= 2:
            dinner_config["usage_hours"] = np.array(dinner_config["usage_hours"]) * 1.75
            breakfast_config["usage_hours"] = np.array(breakfast_config["usage_hours"]) * 1.75
            lunch_config["usage_hours"] = np.array(lunch_config["usage_hours"]) * 1.75

        if max_cook_boxes >= 1.5:
            dinner_config["usage_hours"] = np.array(dinner_config["usage_hours"]) * 1.5
            breakfast_config["usage_hours"] = np.array(breakfast_config["usage_hours"]) * 1.5
            lunch_config["usage_hours"] = np.array(lunch_config["usage_hours"]) * 1.5

        elif max_cook_boxes >= 1.1:
            dinner_config["usage_hours"] = np.array(dinner_config["usage_hours"]) * 1.25
            breakfast_config["usage_hours"] = np.array(breakfast_config["usage_hours"]) * 1.25
            lunch_config["usage_hours"] = np.array(lunch_config["usage_hours"]) * 1.25

        elif max_cook_boxes <= 0.5:
            dinner_config["usage_hours"] = np.array(dinner_config["usage_hours"]) * 0.75
            breakfast_config["usage_hours"] = np.array(breakfast_config["usage_hours"])  * 0.75
            lunch_config["usage_hours"] = np.array(lunch_config["usage_hours"]) * 0.75

        elif max_cook_boxes <= 0.8:
            dinner_config["usage_hours"] = np.array(dinner_config["usage_hours"]) * 0.5
            breakfast_config["usage_hours"] = np.array(breakfast_config["usage_hours"]) * 0.5
            lunch_config["usage_hours"] = np.array(lunch_config["usage_hours"]) * 0.5

    cooking_config.update({
        "breakfast_config": breakfast_config,
        "lunch_config": lunch_config,
        "dinner_config": dinner_config,
        "general_config": general_config,
    })

    return cooking_config
