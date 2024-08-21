
"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Master file for itemization pipeline
"""

# Import python packages

import copy
import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config2 import get_inf_config2
from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config

from python3.itemization.aer.functions.get_config import get_hybrid_config

from python3.itemization.aer.raw_energy_itemization.residual_analysis.config.get_residual_config import get_residual_config


def allot_boxes(item_input_object, item_output_object, appliance_list, output_data, box_tou, logger):

    """
    Prepare hybrid input object

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        appliance_list            (np.ndarray): appliance list
        output_data               (np.ndarray): true disagg output data
        box_tou                   (np.ndarray): tou of detected boxes in residual data
        logger_pass               (dict)      : Contains base logger and logging dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    activity_curve = item_input_object.get("weekday_activity_curve")

    activity_curve = (activity_curve - np.percentile(activity_curve, 3)) / (np.percentile(activity_curve, 97) - np.percentile(activity_curve, 3))

    samples_per_hour = int(output_data.shape[2] / Cgbdisagg.HRS_IN_DAY)

    pilot_level_config = item_input_object.get('pilot_level_config')
    wh_config = get_inf_config().get('wh')

    config = get_residual_config(samples_per_hour).get('stat_app_box_config')

    # default maximum box count in a week

    max_ld_boxes = config.get('max_ld_boxes')
    max_wh_boxes = config.get('max_wh_boxes')
    max_cook_boxes = config.get('max_cook_boxes')
    max_box_amp = config.get('max_box_amp')
    max_box_len = config.get('max_box_len')

    seq_label = seq_config.SEQ_LABEL
    seq_len = seq_config.SEQ_LEN

    pilot = item_input_object.get('config').get('pilot_id')

    # update box count based on lifestyle and config of the user
    # for example if a user has higher cooking for the region, probably the user uses larger nuber of cooking appliances
    # thus we shud give high cooking boxes as an output

    max_cook_boxes, max_ld_boxes = update_box_count_based_on_lifestyle(item_output_object, max_cook_boxes, max_ld_boxes)
    max_cook_boxes, max_ld_boxes = update_box_count_based_on_config(item_input_object, pilot_level_config, max_cook_boxes, max_ld_boxes)

    config = get_inf_config2(item_input_object, samples_per_hour).get("ld")

    # update box count based on ld app profile information
    # and deterime all possible combination of length and cons level for laundry box consumption of the user

    max_ld_boxes, min_amp, max_amp = update_ld_box_count_based_on_app_prof(item_input_object, config, samples_per_hour, max_ld_boxes)

    ld_region = determine_ld_region(item_input_object, samples_per_hour, config, min_amp, max_amp)

    config = get_inf_config2(item_input_object, samples_per_hour).get("wh")

    # and deterime all possible combination of length and cons level for wh box consumption of the user

    wh_region, max_wh_boxes = determine_wh_region(samples_per_hour, config, output_data, appliance_list, max_wh_boxes)

    config = get_inf_config2(item_input_object, samples_per_hour).get("cook")

    # update box count based on cooking app profile information
    # and deterime all possible combination of length and cons level for cooking box consumption of the user

    max_cook_boxes, min_amp, max_amp = update_cook_box_count_based_on_app_prof(item_input_object, config, samples_per_hour, max_cook_boxes)

    cook_region = deterimine_cook_region(samples_per_hour, config, min_amp, max_amp)

    add_wh = item_output_object['box_dict']['hybrid_wh']

    if not add_wh:
        wh_region[:, :] = 0

    box_seq = item_output_object.get("box_dict").get("box_seq_wh")

    # fetching box wise amplitude

    box_amp = copy.deepcopy(box_seq[:, 4])
    box_amp = ((box_amp * samples_per_hour) / 10).astype(int)

    stat_app_box_wise_score = np.zeros((len(box_amp), 3))

    length = copy.deepcopy(box_seq[:, seq_len].astype(int))
    length[box_seq[:, seq_label] == 0] = 0

    # removing boxes with higher consumption and higher box length

    box_amp[box_amp >= max_box_amp/10] = 0
    length[length > max_box_len * samples_per_hour - 1] = 0

    # initialize box score for each appliances
    # ie chances of a box activity belonging to a particular appliance

    stat_app_box_wise_score[:, 0] = wh_region[length, box_amp]
    stat_app_box_wise_score[:, 1] = ld_region[length, box_amp]
    stat_app_box_wise_score[:, 2] = cook_region[length, box_amp]

    stat_app_box_wise_score = np.divide(stat_app_box_wise_score, np.sum(stat_app_box_wise_score, axis=1)[:, None])
    stat_app_box_wise_score = np.nan_to_num(stat_app_box_wise_score)

    # updating box wise score based on domain knowledge
    stat_app_box_wise_score, box_days, box_start = update_box_score_based_on_domain_knowledge(samples_per_hour, item_input_object, box_tou, stat_app_box_wise_score, box_seq)

    stat_app_box_wise_score = update_box_score_based_on_activity(box_days, box_start, item_input_object, stat_app_box_wise_score, activity_curve)

    # initialize max activity boxes per week for each appliance
    max_cook_boxes = int(max_cook_boxes)
    max_wh_boxes = int(max_wh_boxes)
    max_ld_boxes = int(max_ld_boxes)

    wh_disagg = output_data[appliance_list.index("wh") + 1]

    # update max wh box count based on wh disagg output

    max_wh_boxes = update_box_count_based_on_disagg(item_input_object, pilot, wh_config, max_wh_boxes, wh_disagg)

    # determine alloted appliance for each box based on the distribution of scores
    stat_app_box_wise_score = allot_box_score(wh_disagg, stat_app_box_wise_score, box_days, max_wh_boxes, max_ld_boxes, max_cook_boxes)

    return stat_app_box_wise_score


def update_box_count_based_on_lifestyle(item_output_object, max_cook_boxes, max_ld_boxes):

    """
    updates box count based on user lifestyle

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        max_cook_boxes            (int)       : max cooking boxes in a week
        max_ld_boxes              (int)       : max laundry boxes in a week

    Returns:
        max_cook_boxes            (int)       : max cooking boxes in a week
        max_ld_boxes              (int)       : max laundry boxes in a week
    """

    config = get_residual_config().get('stat_app_box_config')

    diff_of_box_count_for_office_goer_user = config.get('diff_of_box_count_for_office_goer_user')

    # decrease box count if the user is office goer

    if item_output_object.get('occupants_profile').get('occupants_features')[item_output_object.get('occupants_profile').get('office_goer_index')]:
        max_ld_boxes = max_ld_boxes - diff_of_box_count_for_office_goer_user

    # increase box count if the user is stay at home

    if item_output_object.get('occupants_profile').get('occupants_features')[item_output_object.get('occupants_profile').get('stay_at_home_index')]:
        max_ld_boxes = max_ld_boxes + diff_of_box_count_for_office_goer_user

    return max_cook_boxes, max_ld_boxes


def update_box_count_based_on_config(item_input_object, pilot_level_config, max_cook_boxes, max_ld_boxes):

    """
    updates box count based on pilot config

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        pilot_level_config        (dict)      : pilot config
        max_cook_boxes            (int)       : max cooking boxes in a week
        max_ld_boxes              (int)       : max laundry boxes in a week

    Returns:
        max_cook_boxes            (int)       : max cooking boxes in a week
        max_ld_boxes              (int)       : max laundry boxes in a week
    """

    config = get_residual_config().get('stat_app_box_config')

    diff_of_box_count_for_dishwasher = config.get('diff_of_box_count_for_dishwasher')
    diff_of_box_count_for_drier = config.get('diff_of_box_count_for_drier')
    cons_bucket = config.get('cons_bucket')
    ld_scaling_factor = config.get('ld_scaling_factor')
    cook_scaling_factor = config.get('cook_scaling_factor')

    hybrid_config = get_hybrid_config(item_input_object.get("pilot_level_config"))

    # removing few ld boxes if dishwasher belongs to cooking

    if hybrid_config.get("dishwash_cat") == "cook":
        max_ld_boxes = max_ld_boxes - diff_of_box_count_for_dishwasher
        max_cook_boxes = max_cook_boxes + diff_of_box_count_for_dishwasher

    # removing few ld boxes if drier is absent

    if "6" in pilot_level_config.get('ld_config').get("drop_app"):
        max_ld_boxes = max_ld_boxes - diff_of_box_count_for_drier

    if ("31" in pilot_level_config.get('ld_config').get("drop_app")) or (
            "33" in pilot_level_config.get('ld_config').get("drop_app")):
        if hybrid_config.get("dishwash_cat") == "cook":
            max_cook_boxes = max_cook_boxes - (diff_of_box_count_for_dishwasher-diff_of_box_count_for_drier)
        else:
            max_ld_boxes = max_ld_boxes + diff_of_box_count_for_dishwasher

    # updating ld boxes based on pilot config

    ld_idx = hybrid_config.get("app_seq").index('ld')
    scale_cons = hybrid_config.get("scale_app_cons")[ld_idx]
    cons_factor = hybrid_config.get("scale_app_cons_factor")[ld_idx]

    if scale_cons:
        max_ld_boxes = max_ld_boxes * (ld_scaling_factor[np.digitize(cons_factor, cons_bucket)])

    cook_idx = hybrid_config.get("app_seq").index('cook')
    scale_cons = hybrid_config.get("scale_app_cons")[cook_idx]
    cons_factor = hybrid_config.get("scale_app_cons_factor")[cook_idx]

    if scale_cons:
        max_cook_boxes = max_cook_boxes * (cook_scaling_factor[np.digitize(cons_factor, cons_bucket)])

    return max_cook_boxes, max_ld_boxes


def update_ld_box_count_based_on_app_prof(item_input_object, inf_config, samples_per_hour, max_ld_boxes):

    """
    updates box count based on app profile

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        config                    (dict)      : config dict
        samples_per_hour          (int)       : samples in an hour
        max_ld_boxes              (int)       : max laundry boxes in a week

    Returns:
        min_amp                   (int)       : min laundry amplitude
        max_amp                   (int)       : max laundry amplitude
        max_ld_boxes              (int)       : max laundry boxes in a week
    """

    occ_count = 1

    if item_input_object.get('home_meta_data').get('numOccupants') is not None and item_input_object.get(
            'home_meta_data').get('numOccupants') > 0:
        occ_count = item_input_object.get('home_meta_data').get('numOccupants')

    config = get_residual_config(samples_per_hour, occ_count).get('stat_app_box_config')

    occ_based_ld_scaling_factor = config.get('occ_based_ld_scaling_factor')
    min_ld_amp = config.get('min_ld_amp')
    ld_amp_buffer = config.get('ld_amp_buffer')
    ld_amp_cap = config.get('ld_amp_cap')

    if not item_input_object.get("appliance_profile").get("default_laundry_flag"):
        ld_app_count = item_input_object.get("appliance_profile").get("laundry")
        ld_app_type = item_input_object.get("appliance_profile").get("laundry_type")
        appliance_consumption = inf_config.get("ld_amp")

        appliance_consumption[2] = np.fmax(ld_amp_cap, appliance_consumption[2])

        ld_app_count[ld_app_type == 0] = ld_app_count[ld_app_type == 0] * 0
        appliance_consumption = np.dot(appliance_consumption, ld_app_count)

        if ld_app_count[2]:
            min_amp = np.fmax(min_ld_amp, appliance_consumption / samples_per_hour - ((inf_config.get("ld_amp")[2]-300) / samples_per_hour))
            min_amp = min(min_amp, inf_config.get("ld_min_amp") / samples_per_hour)
        else:
            min_amp = np.fmax(min_ld_amp, appliance_consumption / samples_per_hour - ld_amp_buffer)
            min_amp = min(min_amp, inf_config.get("ld_min_amp") / samples_per_hour)

        max_amp = appliance_consumption / samples_per_hour + ld_amp_buffer

        max_ld_boxes = (max_ld_boxes / item_input_object.get("appliance_profile").get("default_laundry_count").sum()) * np.sum(ld_app_count)

        max_ld_boxes = max_ld_boxes + 5

    else:
        min_amp = inf_config.get("ld_min_amp") / samples_per_hour
        max_amp = inf_config.get("ld_max_amp") / samples_per_hour

        max_amp = max_amp * occ_based_ld_scaling_factor

    return max_ld_boxes, min_amp, max_amp


def deterimine_cook_region(samples_per_hour, cook_config, min_amp, max_amp):

    """
    initialize region in which cooking boxes could be present

    Parameters:
        samples_per_hour          (int)       : samples in an hour
        cook_config               (dict)      : config dict
        min_amp                   (int)       : min cooking amplitude
        max_amp                   (int)       : max cooking amplitude
    Returns:

        cooking_region            (np.ndarray): region in which cooking boxes could be present
    """

    config = get_residual_config(samples_per_hour).get('stat_app_box_config')
    max_box_amp = int(config.get('max_box_amp')/10)
    max_box_len = config.get('max_box_len')

    cook_region1 = np.zeros((max_box_len * samples_per_hour, max_box_amp))
    cook_region2 = np.zeros((max_box_len * samples_per_hour, max_box_amp))

    min_len = max(1, int(cook_config.get("cook_min_len") * samples_per_hour))
    max_len = int(cook_config.get("cook_max_len") * samples_per_hour)

    min_cap = int((min_amp * samples_per_hour) / 10)
    max_cap = int((max_amp * samples_per_hour) / 10)

    cook_region1[min_len: max_len + 1] = 1
    cook_region2[:, min_cap: max_cap] = 1
    cook_region = np.logical_and(cook_region1, cook_region2)

    return cook_region


def determine_ld_region(item_input_object, samples_per_hour, ld_config, min_amp, max_amp):
    """
    initialize region in which laundry boxes could be present

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        samples_per_hour          (int)       : samples in an hour
        min_amp                   (int)       : min cooking amplitude
        max_amp                   (int)       : max cooking amplitude
    Returns:

        ld_region            (np.ndarray): region in which cooking boxes could be present
    """

    config = get_residual_config(samples_per_hour).get('stat_app_box_config')
    max_box_amp = int(config.get('max_box_amp')/10)
    max_box_len = config.get('max_box_len')
    min_box_len = config.get('min_box_len')
    indian_pilots = PilotConstants.INDIAN_PILOTS

    ld_region1 = np.zeros((max_box_len * samples_per_hour, max_box_amp))
    ld_region2 = np.zeros((max_box_len * samples_per_hour, max_box_amp))

    max_len = ld_config.get("ld_max_len") * samples_per_hour
    min_len = int(ld_config.get("ld_min_len") * samples_per_hour)
    min_cap = int((min_amp * samples_per_hour) / 10)
    max_cap = int((max_amp * samples_per_hour) / 10)

    pilot = item_input_object.get("config").get("pilot_id")

    if pilot in indian_pilots:
        min_len = int(min_box_len * samples_per_hour)

    if item_input_object.get("appliance_profile").get("drier_present"):
        min_len = int(min_box_len * samples_per_hour)

    ld_region1[min_len: max_len + 1] = 1
    ld_region2[:, min_cap: max_cap] = 1
    ld_region = np.logical_and(ld_region1, ld_region2)

    return ld_region


def determine_wh_region(samples_per_hour, config, output_data, appliance_list, max_wh_boxes):
    """
    initialize region in which WH boxes could be present

    Parameters:
        samples_per_hour          (int)       : samples in an hour
        output_data               (np.ndarray): disagg output
        appliance_list            (list)      : list of appliances
        max_wh_boxes              (int)       : max additional wh boxes in a week
    Returns:
        wh_region                 (np.ndarray): region in which WH boxes could be present
    """

    res_config = get_residual_config(samples_per_hour).get('stat_app_box_config')

    max_box_amp = int( res_config.get('max_box_amp')/10)
    max_box_len = res_config.get('max_box_len')

    wh_region1 = np.zeros((max_box_len * samples_per_hour, max_box_amp))
    wh_region2 = np.zeros((max_box_len * samples_per_hour, max_box_amp))

    disagg = output_data[appliance_list.index("wh") + 1]

    if np.sum(disagg):
        min_cap = max(max_box_amp / samples_per_hour, np.median(disagg[disagg > 0]) - 1000 / samples_per_hour)
        max_cap = np.max(disagg) + 300 / samples_per_hour
        max_len = config.get("wh_disagg_max_len") * samples_per_hour
        min_len = config.get("wh_disagg_min_len") * samples_per_hour

    else:
        min_cap = res_config.get('min_cap_wh')
        max_cap = res_config.get('max_cap_wh')
        max_len = res_config.get('max_len_wh')
        min_len = res_config.get('min_len_wh')

    if np.sum(disagg) == 0:
        max_wh_boxes = res_config.get('max_wh_boxes_for_zero_disagg')

    min_cap = int((min_cap * samples_per_hour) / 10)
    max_cap = int((max_cap * samples_per_hour) / 10)
    min_len = int(min_len)
    max_len = int(max_len)

    wh_region1[min_len: max_len + 1] = 1
    wh_region2[:, min_cap: max_cap] = 1
    wh_region = np.logical_and(wh_region1, wh_region2)

    return wh_region, max_wh_boxes


def update_cook_box_count_based_on_app_prof(item_input_object, inf_config, samples_per_hour, max_cook_boxes):

    """
    updates box count based on app profile

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        samples_per_hour          (int)       : samples in an hour
        max_cook_boxes              (int)     : max laundry boxes in a week

    Returns:
        min_amp                   (int)       : min cooking amplitude
        max_amp                   (int)       : max cooking amplitude
        max_cook_boxes              (int)     : max laundry boxes in a week
    """

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    occ_count = 1

    if item_input_object.get('home_meta_data').get('numOccupants') is not None and item_input_object.get(
            'home_meta_data').get('numOccupants') > 0:
        occ_count = item_input_object.get('home_meta_data').get('numOccupants')

    config = get_residual_config(samples_per_hour, occ_count).get('stat_app_box_config')

    occ_based_cook_scaling_factor = config.get('occ_based_cook_scaling_factor')

    if not item_input_object.get("appliance_profile").get("default_cooking_flag"):

        # app profile is present

        cooking_app_count = copy.deepcopy(item_input_object.get("appliance_profile").get("cooking"))
        cooking_app_type = item_input_object.get("appliance_profile").get("cooking_type")
        appliance_consumption = inf_config.get("cooking_amp")

        if str(np.nan_to_num(item_input_object.get('pilot_level_config').get('cook_config').get('type'))) == 'GAS':
            cooking_app_count[cooking_app_type == 2] = cooking_app_count[cooking_app_type == 2] * 2
            cooking_app_count[cooking_app_type == 0] = item_input_object.get("appliance_profile").get("default_cooking_count")[cooking_app_type == 0]
        else:
            cooking_app_count[cooking_app_type == 0] = cooking_app_count[cooking_app_type == 0] * 0

        appliance_consumption = np.dot(appliance_consumption, cooking_app_count)

        min_amp = np.fmax(150/samples_per_hour, appliance_consumption/samples_per_hour - 3500/samples_per_hour)
        max_amp = appliance_consumption/samples_per_hour + 500/samples_per_hour

        app_profile = item_input_object.get("app_profile").get(31)

        if hybrid_config.get("dishwash_cat") == "cook":
            if app_profile is not None:
                app_profile = app_profile.get("number", 0)
            else:
                app_profile = 0

        else:
            app_profile = 0

        cooking_app_count = cooking_app_count.sum() + app_profile * 2

        max_cook_boxes = (max_cook_boxes / item_input_object.get("appliance_profile").get("default_cooking_count").sum()) * cooking_app_count

    else:

        # taking default params since app profile is absent

        min_amp = inf_config.get("cook_min_amp") / samples_per_hour
        max_amp = inf_config.get("cook_max_amp") / samples_per_hour

        max_amp = max_amp * occ_based_cook_scaling_factor

    return max_cook_boxes, min_amp, max_amp


def update_box_count_based_on_disagg(item_input_object, pilot, wh_config, max_wh_boxes, wh_disagg):

    """
    updates WH box count based on WH disagg output

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        pilot                     (int)       : pilot id
        wh_config                 (dict)      : wh config
        max_wh_boxes              (int)       : max WH boxes in a week
        wh_disagg                 (np.ndarray): wh disagg

    Returns:
        max_wh_boxes              (int)       : max WH boxes in a week
    """

    if (item_input_object.get('item_input_params').get('swh_hld') == 0) and \
            (item_input_object.get("item_input_params").get("final_fat_pulse") is not None):
        fat_pulse_tou = item_input_object.get("item_input_params").get("final_fat_pulse") > 0
        wh_seq = find_seq(fat_pulse_tou.flatten(), np.zeros(fat_pulse_tou.size), np.zeros(fat_pulse_tou.size))

        count = (np.sum(wh_seq[:, 0] > 0) / len(wh_disagg)) * Cgbdisagg.DAYS_IN_WEEK
        max_wh_boxes = int(min(max_wh_boxes, count * 3))

        if pilot in wh_config.get("all_year_wh"):
            max_wh_boxes = 100

    if item_input_object.get("item_input_params").get("timed_wh_user"):
        max_wh_boxes = 0

    return max_wh_boxes


def update_box_score_based_on_activity(box_days, box_start, item_input_object, box_score, activity_curve):

    """
    updates scores for each detected box based on the tou and activity score at that time
    these scores denotes the chances of a box being in cooking, laundry or WH category

    Parameters:
        box_start                 (np.ndarray): start index of the detected boxes
        item_input_object         (dict)      : Dict containing all inputs
        box_score                 (np.ndarray): box wise score for all the detected boxes
        activity_curve            (np.ndarray): activity profile of the user

    Returns:
        box_score                 (np.ndarray): updated box wise score for all the detected boxes
    """

    config = get_residual_config().get('stat_app_box_config')

    pilot = item_input_object.get("config").get("pilot_id")

    act_curve_thres = config.get('act_curve_thres')

    if pilot in PilotConstants.INDIAN_PILOTS:
        box_score[:, 2] = box_score[:, 2] * 0.2

    inactive_hours = np.where(activity_curve < act_curve_thres)[0]
    box_score[np.isin(box_start, inactive_hours), 1] = box_score[np.isin(box_start, inactive_hours), 1] * 0.3

    inactive_hours = np.where(activity_curve < act_curve_thres)[0]
    box_score[np.isin(box_start, inactive_hours), 0] = box_score[np.isin(box_start, inactive_hours), 0] * 0

    inactive_hours = np.where(activity_curve < act_curve_thres)[0]
    box_score[np.isin(box_start, inactive_hours), 2] = box_score[np.isin(box_start, inactive_hours), 2] * 0

    inactive_hours = np.where(activity_curve < (np.round(act_curve_thres-0.2, 1)))[0]
    box_score[np.isin(box_start, inactive_hours), 1] = box_score[np.isin(box_start, inactive_hours), 1] * 0

    return box_score


def allot_box_score(disagg, box_score, box_days, max_wh_boxes, max_ld_boxes, max_cook_boxes):

    """
    Alloting each of the detected boxes to cook/ld/wh categoery based on their scores
    these scores denotes the chances of a box being in cooking, laundry or WH category

    Parameters:
        disagg                    (np.ndarray): wh disagg output
        box_score                 (np.ndarray): box wise score for all the detected boxes
        box_days                  (np.ndarray): day index for each of the detected boxes
        max_ld_boxes              (int)       : max laundry boxes in a week
        max_cook_boxes            (int)       : max cooking boxes in a week
        max_ld_boxes              (int)       : max laundry boxes in a week
    Returns:
        box_score                 (np.ndarray): box wise score for all the detected boxes
    """

    max_app_boxes_list = [max_wh_boxes, max_ld_boxes, max_cook_boxes]

    days_in_a_week = Cgbdisagg.DAYS_IN_WEEK

    for i in range(0, len(disagg) - days_in_a_week, days_in_a_week):

        # iterating over WH/LD/COOK

        for app_idx in [0, 1, 2]:

            # checking the number of boxes that can be added to given app category

            max_app_boxes = max_app_boxes_list[app_idx]

            total_app_boxes = box_score[np.isin(box_days, np.arange(i, i + days_in_a_week)), app_idx] == \
                              np.max(box_score[np.isin(box_days, np.arange(i, i + days_in_a_week))], axis=1)
            total_app_boxes = np.logical_and(total_app_boxes, box_score[
                np.isin(box_days, np.arange(i, i + days_in_a_week)), app_idx] > 0)
            total_app_boxes = np.sum(total_app_boxes)
            total_app_days = np.isin(box_days, np.arange(i, i + days_in_a_week)).sum()

            # if the number of boxes is greater than max threshold, extra boxes are removed

            if total_app_boxes > max_app_boxes:
                required_idx = \
                (np.where(np.logical_and(box_score[np.isin(box_days, np.arange(i, i + days_in_a_week)), app_idx] ==
                                         np.max(box_score[np.isin(box_days, np.arange(i, i + days_in_a_week))], axis=1),
                                         box_score[np.isin(box_days, np.arange(i, i + days_in_a_week)), app_idx] > 0)))[0]
                remove_boxes = required_idx[np.arange(0, total_app_boxes, max(1, int(total_app_boxes / (total_app_boxes - max_app_boxes))))]

                picked_app_days = np.ones(total_app_days)
                picked_app_days[remove_boxes] = 0

                box_score[np.isin(box_days, np.arange(i, i + days_in_a_week)), app_idx] = \
                    np.multiply(box_score[np.isin(box_days, np.arange(i, i + days_in_a_week)), app_idx],
                                picked_app_days)

    return box_score


def update_box_score_based_on_domain_knowledge(samples_per_hour, item_input_object, box_tou, box_score, box_seq):

    """
    updates scores for each detected box based characteristics of each appliance category
    these scores denotes the chances of a box being in cooking, laundry or WH category

    Parameters:
        samples_per_hour          (int)       : samples in an hour
        item_input_object         (dict)      : Dict containing all inputs
        box_tou                   (np.ndarray): tou of all detected boxes
        box_score                 (np.ndarray): box wise score for all the detected boxes

    Returns:
        box_score                 (np.ndarray): updated box wise score for all the detected boxes
        box_days                  (np.ndarray): day index for each of the detected boxes
    """

    config = get_residual_config(samples_per_hour).get('stat_app_box_config')

    non_wh_hours = config.get('non_wh_hours')
    cooking_hours = config.get('cooking_hours')
    wh_hours = config.get('wh_hours')
    cons_tou_thres = config.get('cons_tou_thres')
    days_thres_for_swh = config.get('days_thres_for_swh')

    pilot = item_input_object.get("config").get("pilot_id")

    box_tou_day_level = (np.sum(box_tou, axis=0)) / len(box_tou)

    consistent_tou = box_tou_day_level > cons_tou_thres

    consistent_tou_seq = find_seq(consistent_tou > 0, np.zeros_like(consistent_tou), np.zeros_like(consistent_tou))

    for i in range(len(consistent_tou_seq)):
        if consistent_tou_seq[i, 0]:
            consistent_tou[get_index_array(consistent_tou_seq[i, 0] - 1 * samples_per_hour,
                                           consistent_tou_seq[i, 1] + 1 * samples_per_hour,
                                           Cgbdisagg.HRS_IN_DAY * samples_per_hour)] = 1

    consistent_tou = np.where(consistent_tou > 0)[0]

    cooking_hours = cooking_hours.astype(int)
    wh_hours = wh_hours.astype(int)

    box_start = (box_seq[:, 1] % (samples_per_hour * Cgbdisagg.HRS_IN_DAY))
    box_start = box_start.astype(int)

    box_days = (box_seq[:, 1] / (samples_per_hour * Cgbdisagg.HRS_IN_DAY))
    box_days = box_days.astype(int)

    # increasing score of boxes that are consistent throughout the year for wh and cooking
    box_score[np.isin(box_start, consistent_tou), 0] = box_score[np.isin(box_start, consistent_tou), 0] * 2
    box_score[np.isin(box_start, consistent_tou), 2] = box_score[np.isin(box_start, consistent_tou), 2] * 1.5

    box_score[np.isin(box_start, cooking_hours), 2] = copy.deepcopy(box_score[np.isin(box_start, cooking_hours), 2] * 2)

    box_score[np.isin(box_start, wh_hours), 0] = copy.deepcopy(box_score[np.isin(box_start, wh_hours), 0] * 2)

    # not adding wh boxes after afternoon for swh users

    swh_pilots = PilotConstants.SEASONAL_WH_ENABLED_PILOTS

    if len(box_days) <= days_thres_for_swh and pilot in swh_pilots:
        box_score[np.isin(box_start, non_wh_hours), 2] = box_score[np.isin(box_start, non_wh_hours), 2] * 0

    return box_score, box_days, box_start
