
"""
Author - Nisha Agarwal
Date - 4th April 2022
This file contains functions to reduce others output, for billing cycle with more than 15% others
By alloting some extra consumption to cooking/laundry/ent/lighting/ref
"""

# Import python packages

import copy
import numpy as np
from numpy.random import RandomState

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.pilot_constants import PilotConstants

from python3.itemization.init_itemization_config import random_gen_config

from python3.itemization.aer.functions.itemization_utils import get_idx

from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.functions.itemization_utils import rolling_func_along_col

from python3.itemization.aer.raw_energy_itemization.residual_analysis.box_activity_detection_wrapper import box_detection


def get_monthly_others(others, input_data, bc_list, thres=0.1):

    """
    Calculate leftover others at billing cycle that is to be reduced

    Parameters:
        others                    (np.ndarray)    : ts level others
        input_data                (np.ndarray)    : user input data
        bc_list                   (np.ndarray)    : list of billing cycles
        thres                     (float)         : others will be reduced if bc level residual is greater than this fraction

    Returns:
        monthly_cons              (np.ndarray)    : monthly others consumption
    """

    unique_bc = np.unique(bc_list)

    monthly_cons = np.zeros(len(unique_bc))

    for idx, current_bc in enumerate(unique_bc):

        target_days = bc_list == current_bc

        if np.sum(target_days) < 3:
            continue

        monthly_cons[idx] = np.sum(others[target_days]) / np.sum(input_data[target_days])

        monthly_cons[idx] = (monthly_cons[idx] - thres) * np.sum(input_data[target_days])

    monthly_cons = np.fmax(monthly_cons, 0)

    return monthly_cons


def get_monthly_appliance_cons(final_tou_consumption, app, app_list, bc_list, vacation):

    """
    This function calculates appliance consumption at billing cycle level scaled to 30 days

    Parameters:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        app                       (str)           : target appliance
        app_list                  (list)          : list of all appliances
        bc_list                   (np.ndarray)    : billing cycle data of all days
        vacation                  (np.ndarray)    : vacation tags for all days

    Returns:
        monthly_cons              (np.ndarray)    : billing cycle level consumption of the appliance
    """

    unique_bc = np.unique(bc_list)

    monthly_cons = np.zeros(len(unique_bc))

    if np.sum(final_tou_consumption[get_idx(app_list, app)]) <= 0:
        return monthly_cons

    for i in range(len(unique_bc)):

        target_days = bc_list == unique_bc[i]

        if np.sum(target_days) < 3:
            continue

        if np.sum(vacation[target_days]) > 20:
            continue

        if np.sum(np.logical_and(np.logical_not(vacation), target_days)) > 0:
            monthly_cons[i] = \
                np.sum(final_tou_consumption[get_idx(app_list, app)][target_days]) * \
                Cgbdisagg.DAYS_IN_MONTH / np.sum(np.logical_and(np.logical_not(vacation), target_days))

    return monthly_cons


def get_possible_ref(item_output_object, others_output):

    """
    This function extracts the possible consumption that can be alotted to ref, by estimating duty cycle pattern

    Parameters:
        item_output_object         (dict)          : Dict containing all hybrid outputs
        others_output              (np.ndarray))   : ts level others output

    Returns:
        potential_ref_cons          (bool)          : ref type signature extracted from residual data
    """

    temp_val = np.fmax(0, item_output_object.get("hybrid_input_data").get("true_disagg_res"))

    samples = int(others_output.shape[1] / Cgbdisagg.HRS_IN_DAY)

    potential_ref_cons = copy.deepcopy(others_output)

    for i in range(len(others_output[0])):

        idx_arr = get_index_array(i, i + 4*samples, samples*Cgbdisagg.HRS_IN_DAY)

        potential_ref_cons[:, idx_arr] = np.percentile(temp_val[:, idx_arr], 80, axis=1)[:, None]

    potential_ref_cons[:, :] = np.percentile(potential_ref_cons, 20, axis=1)[:, None]

    potential_ref_cons = np.minimum(potential_ref_cons, others_output)

    potential_ref_cons = np.minimum(potential_ref_cons, np.percentile(potential_ref_cons, 95, axis=0)[None, :])

    return potential_ref_cons


def increase_stat_app_in_bc_with_high_others(item_input_object, item_output_object, final_tou_consumption,
                                             processed_input_data, bc_list, app_list, vacation, li_idx, logger):

    """
    This function reduces others output, for billing cycle with more than 15% others

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        processed_input_data      (np.ndarray)    : user input data
        bc_list                   (np.ndarray)    : billing cycle data of all days
        app_list                  (list)          : list of all appliances
        vacation                  (np.ndarray)    : vacation tags for all days
        li_idx                    (int)           : lighting index
        logger                    (logger)        : logger object

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    monthly_others = get_monthly_others(other_cons_arr, processed_input_data, bc_list, thres=0)

    pilot = item_input_object.get("config").get("pilot_id")

    # increasing ref consumption for cases where high ref type consumption was being leftout in disagg residual

    if pilot in PilotConstants.AUSTRALIA_PILOTS:

        high_others = np.ones_like(monthly_others)

        monthly_ld = get_monthly_appliance_cons(final_tou_consumption, 'ref', app_list, bc_list, np.zeros_like(vacation))

        possible_others = copy.deepcopy(other_cons_arr)

        max_ld = np.percentile(monthly_ld, 75) * 1.2

        # calculating potential consumption that can be alloted to ref category

        potential_ref_cons = get_possible_ref(item_output_object, possible_others)

        item_output_object['temp_ref'] = potential_ref_cons

        # adding the above calculated consumption to ref

        final_tou_consumption = reduce_others(final_tou_consumption, 'ref', max_ld, high_others, app_list,
                                              bc_list, np.zeros_like(vacation), potential_ref_cons, monthly_others)

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    monthly_others = get_monthly_others(other_cons_arr, processed_input_data, bc_list)

    samples_per_hour = int(final_tou_consumption.shape[2] /  Cgbdisagg.HRS_IN_DAY)

    user_sleep_hours = get_index_array(item_output_object['profile_attributes']['sleep_time'] * samples_per_hour + samples_per_hour * 2,
                                       item_output_object['profile_attributes']['wakeup_time'] * samples_per_hour - samples_per_hour * 2,
                                       Cgbdisagg.HRS_IN_DAY*samples_per_hour).astype(int)

    # reducing others consumption iteratively, for each appliance
    # this process is stopped once the others output drops below 15% of total energy in all the billing cycles

    if not np.any(monthly_others > 0):
        return final_tou_consumption

    high_others = monthly_others > 0

    # adding others consumption to ref, but keeping in check the existing stability in ref monthly output

    final_tou_consumption = \
        increase_bc_level_ref_cons_to_reduce_others(final_tou_consumption, item_output_object, app_list, bc_list,
                                                    vacation, other_cons_arr, high_others, monthly_others)

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    monthly_others = get_monthly_others(other_cons_arr, processed_input_data, bc_list)

    logger.info('Reducing others value by increasing ref consumption | ')

    if np.any(monthly_others > 0):

        high_others = monthly_others > 0

        # adding others consumption to laundry, but keeping in check the existing stability in laundry monthly output

        appliance = 'ld'

        final_tou_consumption = \
            increase_bc_level_app_cons_to_reduce_others(final_tou_consumption, appliance, app_list, bc_list,
                                                        vacation, 15, other_cons_arr, high_others, monthly_others, user_sleep_hours)

        other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

        monthly_others = get_monthly_others(other_cons_arr, processed_input_data, bc_list)

        logger.info('Reducing others value by increasing laundry consumption | ')

        if np.any(monthly_others > 0):
            high_others = monthly_others > 0

            # adding others consumption to cooking, but keeping in check the existing stability in cooking monthly output

            appliance = 'cook'

            final_tou_consumption = \
                increase_bc_level_app_cons_to_reduce_others(final_tou_consumption, appliance, app_list, bc_list,
                                                            vacation, 15, other_cons_arr, high_others, monthly_others, user_sleep_hours)

            other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

            monthly_others = get_monthly_others(other_cons_arr, processed_input_data, bc_list)

            logger.info('Reducing others value by increasing cooking consumption | ')

            if np.any(monthly_others > 0):

                appliance = 'ent'

                # adding others consumption to ent, but keeping in check the existing stability in ent monthly output

                high_others = monthly_others > 0

                final_tou_consumption = \
                    increase_bc_level_app_cons_to_reduce_others(final_tou_consumption, appliance, app_list, bc_list,
                                                                vacation, 10, other_cons_arr, high_others, monthly_others, user_sleep_hours)

                other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

                monthly_others = get_monthly_others(other_cons_arr, processed_input_data, bc_list)

                logger.info('Reducing others value by increasing ent consumption | ')

                if np.any(monthly_others > 0):
                    high_others = monthly_others > 0

                    # adding others consumption to laundry,
                    # but keeping in check the existing stability in lighting monthly output
                    # and also lighting time of use

                    final_tou_consumption = \
                        increase_bc_level_li_cons_to_reduce_others(final_tou_consumption, li_idx, app_list,
                                                                   bc_list, vacation, 10, other_cons_arr,
                                                                   high_others, monthly_others)

                    logger.info('Reducing others value by increasing li consumption | ')

    return final_tou_consumption


def increase_bc_level_li_cons_to_reduce_others(final_tou_consumption, li_idx, app_list, bc_list, vacation,
                                               max_change, other_cons_arr, high_others, monthly_others):

    """
    This function adds others consumption to li category, to reduce billing cycle level others

   Parameters:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        li_idx                    (int)           : lighting index
        app_list                  (list)          : list of all appliances
        bc_list                   (np.ndarray)    : billing cycle data of all days
        vacation                  (np.ndarray)    : vacation tags for all days
        max_change                (int)           : max deviation allowed from median consumption
        other_cons_arr            (np.ndarray)    : ts level others output
        high_others               (np.ndarray)    : high others billing cycles
        monthly_others            (np.ndarray)    : monthly others output

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    monthly_app_cons = get_monthly_appliance_cons(final_tou_consumption, 'li', app_list, bc_list, vacation)

    possible_others = copy.deepcopy(other_cons_arr)

    perc_for_li_cons = 50

    li_tou = np.sum(final_tou_consumption[li_idx] > 0, axis=0)
    li_tou = li_tou > np.percentile(li_tou, 70)

    possible_others[:, np.logical_not(li_tou)] = 0

    possible_others[possible_others > np.percentile(final_tou_consumption[li_idx], 99)*2] = 0

    min_factor = np.percentile(monthly_app_cons, perc_for_li_cons) / (np.percentile(monthly_app_cons, perc_for_li_cons) + max_change*1000)

    min_factor = 1 / min_factor

    if np.percentile(monthly_app_cons, perc_for_li_cons) == 0:
        min_factor = 0

    max_app_cons = np.percentile(monthly_app_cons, perc_for_li_cons) * min(min_factor, 1.20)

    final_tou_consumption = reduce_others(final_tou_consumption, 'li', max_app_cons, high_others,
                                          app_list, bc_list, vacation, possible_others, monthly_others)
    return final_tou_consumption


def increase_bc_level_ref_cons_to_reduce_others(final_tou_consumption, item_output_object, app_list, bc_list,
                                                vacation, other_cons_arr, high_others, monthly_others):

    """
    This function adds others consumption to ref category, to reduce billing cycle level others

   Parameters:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        item_output_object        (dict)          : Dict containing all hybrid outputs
        app_list                  (list)          : list of all appliances
        bc_list                   (np.ndarray)    : billing cycle data of all days
        vacation                  (np.ndarray)    : vacation tags for all days
        other_cons_arr            (np.ndarray)    : ts level others output
        high_others               (np.ndarray)    : high others billing cycles
        monthly_others            (np.ndarray)    : monthly others output

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    monthly_app_cons = get_monthly_appliance_cons(final_tou_consumption, 'ref', app_list, bc_list, np.zeros_like(vacation))

    possible_others = copy.deepcopy(other_cons_arr)

    max_change = 3

    min_factor = np.median(monthly_app_cons) / (np.median(monthly_app_cons) + max_change * 1000)

    min_factor = 1 / min_factor

    if np.median(monthly_app_cons) == 0:
        min_factor = 0

    max_app_cons = np.median(monthly_app_cons) * min(min_factor, 1.1)

    possible_others = get_possible_ref(item_output_object, possible_others)

    item_output_object['temp_ref'] = possible_others

    final_tou_consumption = reduce_others(final_tou_consumption, 'ref', max_app_cons, high_others, app_list,
                                          bc_list, np.zeros_like(vacation), possible_others, monthly_others)

    return final_tou_consumption


def increase_bc_level_app_cons_to_reduce_others(final_tou_consumption, appliance, app_list, bc_list, vacation,
                                                max_change, other_cons_arr, high_others, monthly_others, sleep_hours):

    """
    This function adds others consumption to appliance category, to reduce billing cycle level others

   Parameters:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        appliance                 (str)           : target appliance(cooking/laundry/entertainment)
        app_list                  (list)          : list of all appliances
        bc_list                   (np.ndarray)    : billing cycle data of all days
        vacation                  (np.ndarray)    : vacation tags for all days
        max_change                (int)           : max deviation allowed from median consumption
        other_cons_arr            (np.ndarray)    : ts level others output
        high_others               (np.ndarray)    : high others billing cycles
        monthly_others            (np.ndarray)    : monthly others output
        sleep_hours               (np.ndarray)    : inactive hours of the user

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    monthly_app_cons = get_monthly_appliance_cons(final_tou_consumption, appliance, app_list, bc_list, vacation)

    possible_others = copy.deepcopy(other_cons_arr)

    # this makes sure that the consumption does not get added during user sleeping hours

    if appliance in ['cook', 'ent', 'ld']:
        possible_others[:, sleep_hours] = 0

    perc_cap_for_consistency = 70

    min_factor = np.percentile(monthly_app_cons, perc_cap_for_consistency) / \
                 (np.percentile(monthly_app_cons, perc_cap_for_consistency) + max_change * 1000)

    min_factor = 1 / min_factor

    if np.percentile(monthly_app_cons, perc_cap_for_consistency) == 0:
        min_factor = 0

    max_app_cons = np.percentile(monthly_app_cons, perc_cap_for_consistency) * min(min_factor, 1.20)

    final_tou_consumption = reduce_others(final_tou_consumption, appliance, max_app_cons, high_others, app_list, bc_list,
                                          vacation, possible_others, monthly_others)

    return final_tou_consumption


def reduce_others(final_tou_consumption, app, max_deviation_allowed, high_others_bc, app_list, bc_list,
                  vacation, possible_others, required_others):

    """
    This function adds others consumption to appliance category, to reduce billing cycle level others

    Parameters:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        app                       (str)           : target appliance
        max_deviation_allowed     (int)           : max consumption deviation allowed from median consumption
        high_others_bc            (np.ndarray)    : list of billing cycles where others output is higher than threshold
        app_list                  (list)          : list of all appliances
        bc_list                   (np.ndarray)    : billing cycle data of all days
        vacation                  (np.ndarray)    : vacation tags for all days
        possible_others           (np.ndarray)    : amount of others that can be alloted to the target appliance
        required_others           (np.ndarray)    : amount of consumption that is to be reduced from each billing cycle

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    unique_bc = np.unique(bc_list)

    index = get_idx(app_list, app)

    possible_others = np.nan_to_num(possible_others)

    if np.sum(final_tou_consumption[get_idx(app_list, app)]) <= 0:
        return final_tou_consumption

    for i in range(len(unique_bc)):

        target_days = bc_list == unique_bc[i]

        if np.sum(target_days) < 3:
            continue

        if not high_others_bc[i]:
            continue

        if np.sum(final_tou_consumption[index][target_days]) == 0:
            continue

        additional_cons = required_others[i]

        additional_cons = min(additional_cons, max_deviation_allowed - (np.sum(final_tou_consumption[get_idx(app_list, app)][target_days]) *
                                                                        Cgbdisagg.DAYS_IN_MONTH / np.sum(np.logical_and(np.logical_not(vacation), target_days))))

        fraction = additional_cons / (np.sum(possible_others[target_days]))

        fraction = min(0.99, fraction)

        if fraction <= 0 or np.isnan(fraction) or (np.sum(possible_others[target_days]) == 0):
            continue

        temp_cons = np.zeros_like(final_tou_consumption[1])

        temp_cons[target_days] = possible_others[target_days] * fraction

        final_tou_consumption[index] = final_tou_consumption[index] + temp_cons

    return final_tou_consumption


def add_box_cons_to_stat_app(pilot, final_tou_consumption, box_cons, processed_input_data, app_list):

    """
    This function performs add leftover box type consumption in residual after itemization into cooking/laundry/entertainment category

    Parameters:
        pilot                     (int)           : pilot id
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        box_cons                  (np.ndarray)    : box type consumption
        processed_input_data      (np.ndarray)    : user input data
        app_list                  (list)          : list of appliances

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    cook_idx = get_idx(app_list, 'cook')
    ld_idx = get_idx(app_list, 'ld')
    ent_idx = get_idx(app_list, 'ent')

    samples_per_hour = int(final_tou_consumption.shape[2] /  Cgbdisagg.HRS_IN_DAY)

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    # preparing cooking and laundry boxes to be added into their respective category
    # This step was added to reduce others consumption that was left out even after complete itemization

    box_cons[box_cons > 0] = np.maximum(box_cons[box_cons > 0], other_cons_arr[box_cons > 0])
    box_cons = np.minimum(box_cons, other_cons_arr)

    total_frac = final_tou_consumption[ld_idx].sum() + final_tou_consumption[ent_idx].sum() + final_tou_consumption[cook_idx].sum()

    ld_frac = final_tou_consumption[ld_idx].sum() / total_frac
    cook_frac = final_tou_consumption[cook_idx].sum() / total_frac
    ent_frac = final_tou_consumption[ent_idx].sum() / total_frac

    day_hours = np.arange(6*samples_per_hour, Cgbdisagg.HRS_IN_DAY*samples_per_hour).astype(int)

    box_cons_ld = copy.deepcopy(box_cons)
    if np.sum(final_tou_consumption[ld_idx]) > 0:
        max_val = np.percentile(final_tou_consumption[ld_idx][final_tou_consumption[ld_idx] > 0], 80)
        box_cons_ld[(box_cons_ld+final_tou_consumption[ld_idx]) < max_val] = 0
    else:
        box_cons_ld[:, :] = 0

    box_cons_cook = copy.deepcopy(box_cons)

    if np.sum(final_tou_consumption[cook_idx]) > 0:
        max_val = np.percentile(final_tou_consumption[cook_idx][final_tou_consumption[cook_idx] > 0], 70)
        box_cons_ld[(box_cons_ld+final_tou_consumption[cook_idx]) < max_val] = 0
    else:
        box_cons_ld[:, :] = 0

    final_tou_consumption[ld_idx][:, day_hours] = final_tou_consumption[ld_idx][:, day_hours] + box_cons_ld[:, day_hours] * ld_frac
    final_tou_consumption[cook_idx][:, day_hours] = final_tou_consumption[cook_idx][:, day_hours] + box_cons_cook[:, day_hours] * cook_frac
    final_tou_consumption[ent_idx][:, day_hours] = final_tou_consumption[ent_idx][:, day_hours] + box_cons[:, day_hours] * ent_frac

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    box_label, box_cons_ent, box_seq = \
        box_detection(pilot, other_cons_arr, np.fmax(0, other_cons_arr), np.zeros_like(other_cons_arr),
                      min_amp=200 / samples_per_hour, max_amp=10000 / samples_per_hour, min_len=1,
                      max_len=2 * samples_per_hour, detect_wh=1)

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    box_cons_ent[box_cons_ent > 0] = np.maximum(box_cons_ent[box_cons_ent > 0], other_cons_arr[box_cons_ent > 0])
    box_cons_ent = np.minimum(box_cons_ent, other_cons_arr)

    box_cons_ld = copy.deepcopy(box_cons_ent)

    if np.sum(final_tou_consumption[ld_idx]) > 0:
        max_val = np.percentile(final_tou_consumption[ld_idx][final_tou_consumption[ld_idx] > 0], 80)
        box_cons_ld[(box_cons_ld+final_tou_consumption[ld_idx]) < max_val] = 0
    else:
        box_cons_ld[:, :] = 0

    box_cons_cook = copy.deepcopy(box_cons_ent)

    if np.sum(final_tou_consumption[cook_idx]) > 0:
        max_val = np.percentile(final_tou_consumption[cook_idx][final_tou_consumption[cook_idx] > 0], 70)
        box_cons_ld[(box_cons_ld+final_tou_consumption[cook_idx]) < max_val] = 0
    else:
        box_cons_ld[:, :] = 0

    total_frac = final_tou_consumption[ld_idx].sum() + final_tou_consumption[ent_idx].sum() + final_tou_consumption[cook_idx].sum()

    ld_frac = final_tou_consumption[ld_idx].sum() / total_frac
    cook_frac = final_tou_consumption[cook_idx].sum() / total_frac
    ent_frac = final_tou_consumption[ent_idx].sum() / total_frac

    day_start_hour = 6

    day_hours = np.arange(day_start_hour * samples_per_hour, Cgbdisagg.HRS_IN_DAY * samples_per_hour).astype(int)

    final_tou_consumption[ld_idx][:, day_hours] = final_tou_consumption[ld_idx][:, day_hours] + box_cons_ld[:, day_hours] * ld_frac
    final_tou_consumption[cook_idx][:, day_hours] = final_tou_consumption[cook_idx][:, day_hours] + box_cons_cook[:, day_hours] * cook_frac
    final_tou_consumption[ent_idx][:, day_hours] = final_tou_consumption[ent_idx][:, day_hours] + box_cons_ent[:, day_hours] * ent_frac

    return final_tou_consumption, box_cons


def pick_potential_ent_cons_from_residual(item_input_object, final_tou_consumption, appliance_list, other_cons_arr, sleep_hours):

    """
    This function ensure max ts level consumption. This function is added to avoid any high consumption
     being added after itemization

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)
        appliance_list              (list)          : list of target appliances
        other_cons_arr              (np.ndarray)    : hybrid v2 residual
        sleep_hours                 (np.ndarray)    : sleep hours of the user

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    if len(final_tou_consumption[1]) < 100:
        return final_tou_consumption

    seed = RandomState(random_gen_config.seed_value)

    sleep_hours = np.logical_not(sleep_hours.astype(bool))

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')
    ent_idx = np.where(appliance_list == 'ent')[0][0]

    activity_curve_thres = 0.2
    max_ent_cons = 300
    inactive_hours = np.arange(2 * samples_per_hour, 6 * samples_per_hour + 1)

    consumption_diff = np.zeros_like(final_tou_consumption[ent_idx])

    # calculating potential entertainment consumption by comparing day baseload consumption with consumption in low activity period
    # for every chunk of days

    days_in_month = Cgbdisagg.DAYS_IN_MONTH

    for i in range(0, len(other_cons_arr) - days_in_month, 10):
        consumption_diff[i:i + days_in_month] = np.mean(other_cons_arr[i:i + days_in_month], axis=0) - \
                                                np.percentile(np.mean(other_cons_arr[i:i + days_in_month][:, sleep_hours], axis=0), 40)

    consumption_diff = rolling_func_along_col(consumption_diff, int(samples_per_hour * 1.5))

    consumption_diff = np.fmax(0, consumption_diff)

    consumption_diff = np.median(consumption_diff, axis=0)

    ent_cons = np.zeros_like(final_tou_consumption[0])

    consumption_diff[sleep_hours] = 0

    consumption_diff[inactive_hours] = 0

    potential_ent_points = np.logical_and(consumption_diff > 20 / samples_per_hour, consumption_diff < 5000 / samples_per_hour)

    ent_cons[:, potential_ent_points] = consumption_diff[potential_ent_points][None, :]

    ent_cons = np.fmin(max_ent_cons / samples_per_hour, ent_cons)

    # Calculating weekday and weekend entertainment consumption separately

    weekend_block = copy.deepcopy(ent_cons)

    dow = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_DOW_IDX, :, 0]
    activity_curve = item_input_object.get("weekday_activity_curve")
    activity_curve = (activity_curve - np.percentile(activity_curve, 3)) / (np.percentile(activity_curve, 97) - np.percentile(activity_curve, 3))
    weekend_days = np.logical_or(dow == 1, dow == 7)

    weekend_block[:, activity_curve < activity_curve_thres] = 0
    weekend_block[np.logical_not(weekend_days)] = 0

    weekday_block = copy.deepcopy(ent_cons)

    activity_curve = item_input_object.get("weekend_activity_curve")
    activity_curve = (activity_curve - np.percentile(activity_curve, 3)) / (np.percentile(activity_curve, 97) - np.percentile(activity_curve, 3))

    weekday_block[:, activity_curve < activity_curve_thres] = 0
    weekday_block[weekend_days] = 0

    ent_cons = weekend_block + weekday_block

    if np.sum(final_tou_consumption[ent_idx]):
        final_tou_consumption[ent_idx] = final_tou_consumption[ent_idx] + \
                                         np.fmax(0, np.minimum(other_cons_arr, np.multiply(ent_cons, seed.normal(0.9, 0.1, ent_cons.shape))))

    return final_tou_consumption


def add_leftover_potential_boxes_to_hvac(final_tou_consumption, item_input_object, pilot, processed_input_data,
                                         samples_per_hour, cooling_idx, heat_idx):

    """
    This function performs add leftover box type consumption in residual after itemization into HVAC category,
     if the tou/amp matches with original HVAC output

    Parameters:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        item_input_object         (dict)          : Dict containing all hybrid inputs
        pilot                     (int)           : pilot id
        processed_input_data      (np.ndarray)    : user input data
        samples_per_hour          (int)           : samples in an hour
        cooling_idx               (int)           : cooling index
        heat_idx                  (int)           : heating index

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    lower_perc_cap = 20
    upper_perc_cap = 97

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    box_label, box_cons_in_residual, box_seq = \
        box_detection(pilot, other_cons_arr, np.fmax(0, other_cons_arr), np.zeros_like(other_cons_arr),
                      min_amp=500 / samples_per_hour, max_amp=10000 / samples_per_hour, min_len=1,
                      max_len=3 * samples_per_hour, detect_wh=1)

    box_cons = copy.deepcopy(box_cons_in_residual)
    box_cons_hvac = copy.deepcopy(box_cons_in_residual)

    temp_cool = copy.deepcopy(final_tou_consumption[cooling_idx])

    # picking potential boxes for cooling category

    if item_input_object.get("item_input_params").get("ao_cool") is not None:
        temp_cool = temp_cool - item_input_object.get("item_input_params").get("ao_cool")
        temp_cool = np.fmax(0, temp_cool)

    # picking boxes that in neighbourhood of existing on demand hvac consumption

    box_cons[np.logical_and(temp_cool == 0, np.logical_and(np.roll(temp_cool, 1, axis=1)==0, np.roll(temp_cool, -1, axis=1)==0))] = 0
    box_cons[np.logical_and(temp_cool == 0, np.logical_and(np.roll(temp_cool, 1, axis=0)==0, np.roll(temp_cool, -1, axis=0)==0))] = 0

    # picking boxes with amplitude in the range of existing on demand hvac consumption

    if temp_cool.sum() > 0:
        min_cons = np.percentile(final_tou_consumption[cooling_idx][final_tou_consumption[cooling_idx] > 0], lower_perc_cap)
        max_cons = np.percentile(final_tou_consumption[cooling_idx][final_tou_consumption[cooling_idx] > 0], upper_perc_cap)

        box_cons[(box_cons+temp_cool) > max_cons*1.1] = 0
        box_cons[(box_cons+temp_cool) < min_cons] = 0

        final_tou_consumption[cooling_idx] = box_cons + final_tou_consumption[cooling_idx]

    # picking potential boxes for heating category

    temp_heat = copy.deepcopy(final_tou_consumption[heat_idx])

    if item_input_object.get("item_input_params").get("ao_heat") is not None:
        temp_heat = temp_heat - item_input_object.get("item_input_params").get("ao_heat")
        temp_heat = np.fmax(0, temp_heat)

    # picking boxes that in neighbourhood of existing on demand hvac consumption
    box_cons_hvac[np.logical_and(temp_heat == 0, np.logical_and(np.roll(temp_heat, 1, axis=1)==0, np.roll(temp_heat, -1, axis=1)==0))] = 0
    box_cons_hvac[np.logical_and(temp_heat == 0, np.logical_and(np.roll(temp_heat, 1, axis=0)==0, np.roll(temp_heat, -1, axis=0)==0))] = 0

    # picking boxes with amplitude in the range of existing on demand hvac consumption

    if temp_heat.sum() > 0:
        min_cons = np.percentile(final_tou_consumption[heat_idx][final_tou_consumption[heat_idx] > 0], lower_perc_cap)
        max_cons = np.percentile(final_tou_consumption[heat_idx][final_tou_consumption[heat_idx] > 0], upper_perc_cap)

        box_cons_hvac[(box_cons_hvac + temp_heat) > max_cons*1.1] = 0
        box_cons_hvac[(box_cons_hvac + temp_heat) < min_cons] = 0

        final_tou_consumption[heat_idx] = box_cons_hvac + final_tou_consumption[heat_idx]

    return final_tou_consumption
