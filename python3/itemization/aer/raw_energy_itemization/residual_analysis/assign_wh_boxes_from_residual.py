
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

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.functions.hsm_utils import check_validity_of_hsm

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config

from python3.itemization.aer.raw_energy_itemization.residual_analysis.detect_hybrid_wh import detect_hybrid_wh


def update_hsm(item_input_object, item_output_object, disagg_cons, hybrid_wh):

    """
    Update wh hsm with hybrid wh attributes

    Parameters:
        item_input_object           (dict)             : Dict containing all hybrid inputs
        item_output_object          (dict)             : Dict containing all hybrid outputs
        disagg_cons                 (np.ndarray)       : wh disagg output
        hybrid_wh                   (int)              : flag that represents whether WH is added from hybrid

    Returns:
        item_output_object          (dict)             : Dict containing all hybrid outputs
    """

    created_hsm = dict({
        'item_tou': np.zeros(len(disagg_cons[0])),
        'item_hld': 0,
        'item_type': 0,
        'item_amp': 0
    })

    # updating HSM in cases where WH is being added from hybrid v2

    if hybrid_wh:
        created_hsm['item_tou'] = np.ones(len(disagg_cons[0]))
        created_hsm['item_amp'] = 1500/int(len(disagg_cons[0]) / Cgbdisagg.HRS_IN_DAY)
        created_hsm['item_hld'] = 1
        created_hsm['item_type'] = 2
    else:
        created_hsm['item_tou'] = np.zeros(len(disagg_cons[0]))

    post_hsm_flag = item_input_object.get('item_input_params').get('post_hsm_flag')

    # updating HSM for historical/incremental runs

    wh_hsm_key_present = post_hsm_flag and (item_output_object.get('created_hsm').get('wh') is None)

    if wh_hsm_key_present:
        item_output_object['created_hsm']['wh'] = {
            'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': dict()
        }

    wh_hsm_key_present = \
        post_hsm_flag and (item_output_object.get('created_hsm').get('wh') is not None) and \
        (item_output_object.get('created_hsm').get('wh').get('attributes') is not None)

    if wh_hsm_key_present:
        item_output_object['created_hsm']['wh']['attributes'].update(created_hsm)

    return item_output_object


def check_wh_addition_bool_based_on_hsm(valid_wh_hsm, add_wh, hsm_wh_amp, item_input_object, item_output_object, logger):

    """
    checks whether WH can be added based on previous run info

    Parameters:
        valid_wh_hsm                (int)              : flag to represent whether WH HSM can be used
        add_wh                      (int)              : default wh addition flag
        hsm_wh_amp                  (int)              : default WH amp
        item_input_object           (dict)             : Dict containing all hybrid inputs
        item_output_object          (dict)             : Dict containing all hybrid outputs
        logger                      (logger)           : logger object

    Returns:
        add_wh                      (int)              : wh addition flag
        hsm_wh_amp                  (int)              : WH amp
    """

    valid_hsm_flag = check_validity_of_hsm(valid_wh_hsm, item_input_object.get("item_input_params").get('wh_hsm'), 'item_hld')

    if not valid_hsm_flag:
        logger.info('WH HSM info is absent | %s', add_wh)
        return add_wh, hsm_wh_amp

    # checking whether WH HLD is true in HSM data

    wh_hsm = item_input_object.get("item_input_params").get('wh_hsm').get('item_hld')
    hsm_wh_hld = 0

    if wh_hsm is not None and isinstance(wh_hsm, list):
        hsm_wh_hld = wh_hsm[0]
    elif wh_hsm is not None:
        hsm_wh_hld = wh_hsm

    # checking whether WH type is storage in HSM data

    wh_hsm = item_input_object.get("item_input_params").get('wh_hsm').get('item_type')
    hsm_wh_type = 2

    if wh_hsm is not None and isinstance(wh_hsm, list):
        hsm_wh_type = wh_hsm[0]
    elif wh_hsm is not None:
        hsm_wh_type = wh_hsm

    # checking WH amplitude in HSM data

    add_wh = hsm_wh_hld

    wh_hsm = item_input_object.get("item_input_params").get('wh_hsm').get('item_amp')

    if wh_hsm is not None and isinstance(wh_hsm, list):
        hsm_wh_amp = wh_hsm[0]
    elif wh_hsm is not None:
        hsm_wh_amp = wh_hsm

    # If Timed WH present in WH HSM data, storage WH wont be addded

    if item_input_object.get("item_input_params").get("timed_wh_user") or hsm_wh_type == 1:
        add_wh = 0
        hsm_wh_amp = 0
        logger.info('Timed WH present in WH HSM data | ')

    app_profile = item_input_object.get("app_profile").get('wh')

    if app_profile is not None:
        app_profile = app_profile.get("number", 1)
    else:
        app_profile = 1

    if add_wh and (not app_profile):
        add_wh = 0
        hsm_wh_amp = 0

        logger.info('WH not being added due to 0 profile users | ')

    logger.info('WH box addition flag based on HSM info and app profile| %s', add_wh)

    item_output_object['box_dict']['hybrid_wh'] = add_wh

    return add_wh, hsm_wh_amp


def get_wh_addition_params(disagg_cons, amp, add_wh, samples_per_hour, item_input_object):

    """
    Prepare hybrid input object

    Parameters:
        disagg_cons               (np.ndarray)       : wh disagg output
        amp                       (int)              : default wh amp
        add_wh                    (int)              : WH addition flag
        samples_per_hour          (int)              : samples in an hour
        item_input_object         (dict)             : Dict containing all inputs

    Returns:
        min_cap                   (int)              : WH min amp
        max_cap                   (int)              : WH max amp
        min_len                   (int)              : WH min len
        max_len                   (int)              : WH max len
    """

    config = get_inf_config(samples_per_hour).get("wh")

    min_wh_amp = config.get('box_add_min_wh_amp')
    min_wh_len = config.get('box_add_min_wh_len')
    min_tankless_wh_amp = config.get('box_add_min_tankless_wh_amp')
    max_wh_amp = config.get('box_add_max_wh_amp')
    max_wh_amp_inc = config.get('box_add_max_wh_amp_inc')
    min_wh_amp_dec = config.get('box_add_min_wh_amp_dec')
    min_wh_for_nonzero_disagg = config.get('box_add_min_wh_for_nonzero_disagg')

    add_wh_in_mtd = add_wh and (item_input_object.get("config").get('disagg_mode') == 'mtd')

    if np.sum(disagg_cons):

        fat_pulse_info_present_flag = \
            (item_input_object.get('item_input_params').get('swh_hld') == 0) and \
            (item_input_object.get("item_input_params").get("final_fat_pulse") is not None) and \
            (item_input_object.get("item_input_params").get("final_fat_pulse").sum() > 0)

        # if disagg is non zero, WH is added based on disagg box amplitude

        if fat_pulse_info_present_flag:
            fat_pulse_tou = item_input_object.get("item_input_params").get("final_fat_pulse") > 0

            disagg_fat_pulse_overlapping_points_present = np.sum(np.logical_and(disagg_cons > 0, fat_pulse_tou > 0)) > 0

            if disagg_fat_pulse_overlapping_points_present:

                min_cap = max(min_wh_for_nonzero_disagg / samples_per_hour,
                              np.median(disagg_cons[np.logical_and(disagg_cons > 0, fat_pulse_tou > 0)]) - min_wh_amp_dec / samples_per_hour)
                max_cap = np.max(disagg_cons) + max_wh_amp_inc / samples_per_hour

            else:
                min_cap = min_wh_amp / samples_per_hour
                max_cap = amp / samples_per_hour
        else:
            min_cap = min_wh_amp / samples_per_hour
            max_cap = amp / samples_per_hour

        max_len = config.get("wh_disagg_max_len") * samples_per_hour
        min_len = config.get("wh_disagg_min_len") * samples_per_hour

    # WH in mtd mode
    elif add_wh_in_mtd:
        min_cap = min_wh_amp / samples_per_hour
        max_cap = (amp + 1000) / samples_per_hour
        max_len = config.get("wh_max_len") * samples_per_hour
        min_len = min_wh_len * samples_per_hour

    # adding WH based on previous run info
    elif add_wh:
        min_cap = min_wh_amp / samples_per_hour
        max_cap = amp / samples_per_hour
        max_len = config.get("wh_max_len") * samples_per_hour
        min_len = min_wh_len * samples_per_hour

    # adding WH from hybrid v2
    else:
        min_cap = min_wh_amp / samples_per_hour
        max_cap = max_wh_amp / samples_per_hour
        max_len = config.get("wh_max_len") * samples_per_hour
        min_len = min_wh_len * samples_per_hour

    # updating min capp for tankless WH
    if item_input_object.get('item_input_params').get('tankless_wh') > 0:
        min_cap = max(min_cap, min_tankless_wh_amp / samples_per_hour)

    return min_cap, max_cap, min_len, max_len


def allot_wh_boxes(item_input_object, item_output_object, appliance_list, output_data, box_tou, logger):

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

    activity_curve = (activity_curve - np.percentile(activity_curve, 3)) / \
                     (np.percentile(activity_curve, 97) - np.percentile(activity_curve, 3))

    samples_per_hour = int(output_data.shape[2] / Cgbdisagg.HRS_IN_DAY)

    config = get_inf_config(samples_per_hour).get("wh")

    max_wh_boxes = config.get("max_wh_boxes")

    wh_region1 = np.zeros((4 * samples_per_hour, 900))
    wh_region2 = np.zeros((4 * samples_per_hour, 900))

    app_profile = item_input_object.get("app_profile").get('wh')

    disagg = output_data[appliance_list.index("wh")+1]

    if app_profile is not None:
        app_profile = app_profile.get("number", 0)
    else:
        app_profile = 0

    item_output_object['box_dict']['hybrid_wh'] = 0

    hybrid_wh = detect_hybrid_wh(item_input_object, item_output_object, disagg, logger)

    add_wh = (app_profile or np.sum(disagg) or hybrid_wh) and not(item_input_object.get("item_input_params").get("timed_wh_user"))

    item_output_object = update_hsm(item_input_object, item_output_object, disagg, hybrid_wh)

    valid_wh_hsm = item_input_object.get("item_input_params").get('valid_wh_hsm')

    amp = config.get("max_wh_amp")

    # checking whether to add active pulse boxes to hybrid wh

    add_wh, amp = check_wh_addition_bool_based_on_hsm(valid_wh_hsm, add_wh, amp, item_input_object, item_output_object, logger)

    min_cap, max_cap, min_len, max_len = get_wh_addition_params(disagg, amp, add_wh, samples_per_hour, item_input_object)

    logger.info('Min capacity while WH box addition | %s', min_cap)
    logger.info('Max capacity while WH box addition | %s', max_cap)
    logger.info('Min length while WH box addition | %s', min_len)
    logger.info('Max length while WH box addition | %s', max_len)

    logger.info('WH box addition flag | %s', add_wh)

    if np.sum(disagg) == 0:
        max_wh_boxes = config.get("max_wh_boxes_for_hld_change")

    min_cap = int((min_cap * samples_per_hour) / 10)
    max_cap = int((max_cap * samples_per_hour) / 10)
    min_len = max(1, int(min_len))
    max_len = int(max_len)

    wh_region1[min_len: max_len+1] = 1
    wh_region2[:, min_cap: max_cap] = 1
    wh_region = np.logical_and(wh_region1, wh_region2)

    # finding amp  and tou range where wh box can be added

    if not add_wh:
        wh_region[:, :] = 0

    box_seq = item_output_object.get("box_dict").get("box_seq_wh")

    if not item_output_object['box_dict']['hybrid_wh']:
        item_output_object['box_dict']['hybrid_wh'] = hybrid_wh

    box_amp = copy.deepcopy(box_seq[:, 4])

    box_amp = ((box_amp * samples_per_hour) / 10).astype(int)

    box_score = np.zeros((len(box_amp), 3))

    length = copy.deepcopy(box_seq[:, 3].astype(int))
    length[box_seq[:, 0] == 0] = 0
    length[box_seq[:, 0] == 0] = 0

    box_amp[box_amp >= 900] = 0
    length[length > 4*samples_per_hour-1] = 0

    box_score[:, 0] = wh_region[length, box_amp]

    box_score = np.divide(box_score, np.sum(box_score, axis=1)[:, None])

    box_score = np.nan_to_num(box_score)

    box_usage_hours = (np.sum(box_tou, axis=0)) / len(box_tou)

    consistent_tou = box_usage_hours > 0.75

    # update boxes score based on tou

    consistent_tou_seq = find_seq(consistent_tou, np.zeros_like(consistent_tou), np.zeros_like(consistent_tou))

    for i in range(len(consistent_tou_seq)):
        if consistent_tou_seq[i, 0]:
            consistent_tou[get_index_array(consistent_tou_seq[i, 1]-1*samples_per_hour,
                                           consistent_tou_seq[i, 0]+1*samples_per_hour, Cgbdisagg.HRS_IN_DAY*samples_per_hour)] = 1

    consistent_tou = np.where(consistent_tou > 0)[0]

    wh_hours = np.arange(6 * samples_per_hour, 9 * samples_per_hour + 1)

    wh_hours = wh_hours.astype(int)

    box_start = (box_seq[:, 1] % (samples_per_hour * Cgbdisagg.HRS_IN_DAY))
    box_start = box_start.astype(int)

    box_days = (box_seq[:, 1] / (samples_per_hour * Cgbdisagg.HRS_IN_DAY))
    box_days = box_days.astype(int)

    pilot = item_input_object.get("config").get("pilot_id")

    wh_box_idx = 0

    box_score[np.isin(box_start, consistent_tou), wh_box_idx] = box_score[np.isin(box_start, consistent_tou), wh_box_idx] * 2
    box_score[np.isin(box_start, wh_hours), wh_box_idx] = box_score[np.isin(box_start, wh_hours), wh_box_idx] * 2

    inactive_hours = np.where(activity_curve < 0.4)[0]
    box_score[np.isin(box_start, inactive_hours), wh_box_idx] = box_score[np.isin(box_start, inactive_hours), wh_box_idx] * 0.3

    inactive_hours = np.where(activity_curve < 0.3)[0]
    box_score[np.isin(box_start, inactive_hours), wh_box_idx] = box_score[np.isin(box_start, inactive_hours), wh_box_idx] * 0

    max_wh_boxes = int(max_wh_boxes)

    # update max box count based on disagg output

    if (item_input_object.get('item_input_params').get('swh_hld') == 0) and \
            (item_input_object.get("item_input_params").get("final_fat_pulse") is not None) and \
            (disagg.sum() > 0):
        fat_pulse_tou = item_input_object.get("item_input_params").get("final_fat_pulse") > 0
        wh_seq = find_seq(fat_pulse_tou.flatten(), np.zeros(fat_pulse_tou.size), np.zeros(fat_pulse_tou.size))

        count = (np.sum(wh_seq[:, 0] > 0) / len(disagg)) * 7
        max_wh_boxes = int(min(max_wh_boxes, count * 3))

        if pilot in config.get("all_year_wh"):
            max_wh_boxes = config.get("max_wh_boxes_for_hld_change") * 2

    # not adding active pulses for twh

    if item_input_object.get("item_input_params").get("timed_wh_user"):
        max_wh_boxes = 0

    logger.info("Max box count | %s", max_wh_boxes)

    window = 7

    # remove extra wh boxes

    for i in range(0, len(disagg)-window, window):

        total_boxes = box_score[np.isin(box_days, np.arange(i, i+window)), 0] == np.max(box_score[np.isin(box_days, np.arange(i, i+window))], axis=1)
        total_boxes = np.logical_and(total_boxes, box_score[np.isin(box_days, np.arange(i, i+window)), 0] > 0)
        total_boxes = np.sum(total_boxes)
        total_days = np.isin(box_days, np.arange(i, i+window)).sum()

        if total_boxes > max_wh_boxes:
            required_idx = (np.where(np.logical_and(box_score[np.isin(box_days, np.arange(i, i + window)), 0] ==
                                                    np.max(box_score[np.isin(box_days, np.arange(i, i + window))], axis=1),
                                                    box_score[np.isin(box_days, np.arange(i, i + window)), 0] > 0)))[0]

            remove_boxes = required_idx[np.arange(0, total_boxes, int(total_boxes / (total_boxes - max_wh_boxes)))]

            days_with_removed_boxes = np.ones(total_days)
            days_with_removed_boxes[remove_boxes] = 0

            box_score[np.isin(box_days, np.arange(i, i+window)), 0] = np.multiply(box_score[np.isin(box_days, np.arange(i, i+window)), 0], days_with_removed_boxes)

    return box_score
