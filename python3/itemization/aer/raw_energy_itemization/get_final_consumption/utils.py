
"""
Author - Nisha Agarwal
Date - 4th April 2021
Utils functions for calculating 100% itemization module
"""

# Import python packages

import copy
import numpy as np
from numpy.random import RandomState

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.pilot_constants import PilotConstants

from python3.itemization.aer.functions.itemization_utils import find_seq

from python3.itemization.init_itemization_config import random_gen_config

from python3.itemization.aer.functions.get_config import get_hybrid_config

from python3.itemization.initialisations.get_thresholds_config import get_thresholds_config

from python3.itemization.aer.functions.itemization_utils import fill_arr_based_seq_val_for_valid_boxes

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.init_final_item_config import init_final_item_conf


def limit_wh_delta(item_input_object, item_output_object, final_tou_consumption, appliance_list,
                   output_data, processed_input_data, hsm_wh, logger):

    """
    This function is added to limit BC level delta from disagg of HVAC appliances
     (In order to control WH overestimation in hybrid module)

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)
        appliance_list              (list)          : list of appliances
        output_data                 (np.ndarray)    : ts level true disagg output for all appliances
        processed_input_data        (np.ndarray)    : ts level input data
        hsm_wh                      (int)           : wh monthly output in previous run
        logger                      (logger)        : logger object

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    config = init_final_item_conf().get('post_processing_config')

    # This functions the increase of cooling (in hybrid module) to 30%, excluding the detected seasonal signature

    pilot = item_input_object.get("config").get("pilot_id")

    swh_pilot_user = pilot in PilotConstants.SEASONAL_WH_ENABLED_PILOTS

    thres = (-config.get('max_swh_delta')) * (swh_pilot_user) + (-config.get('max_wh_delta')) * (not swh_pilot_user)

    season = copy.deepcopy(item_output_object.get("season"))
    season[season == -0.5] = -1
    season[season == 0.5] = 1

    if item_input_object.get('item_input_params').get('swh_hld') > 0:
        return final_tou_consumption

    # remove new added boxes if delta from disagg for wh is more than 50%

    if int(item_input_object.get("item_input_params").get("timed_wh_user")):

        final_tou_consumption = limit_wh_delta_for_inc(item_input_object, item_output_object, final_tou_consumption,
                                                       appliance_list, output_data, processed_input_data, hsm_wh,
                                                       logger)

        final_tou_consumption = limit_wh_delta_for_mtd(item_input_object, item_output_object, final_tou_consumption,
                                                       appliance_list, output_data, processed_input_data, hsm_wh,
                                                       logger)

        return final_tou_consumption

    wh_idx = np.where(appliance_list == 'wh')[0][0]

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)
    season_list = np.zeros_like(unique_bc)
    cons_list = np.zeros_like(unique_bc)
    unique_bc = unique_bc[counts >= 4]

    for bill_cycle in range(len(unique_bc)):
        season_list[bill_cycle] = np.sign(season[bc_list == unique_bc[bill_cycle]].mean())
        cons_list[bill_cycle] = np.sum(output_data[wh_idx, bc_list == unique_bc[bill_cycle]]) * Cgbdisagg.DAYS_IN_MONTH/np.sum(bc_list == unique_bc[bill_cycle])

    if item_input_object.get('item_input_params').get('run_hybrid_v2_flag'):
        min_cons = item_input_object.get('pilot_level_config').get('wh_config').get('bounds').get('min_cons')
    else:
        min_cons = 5

    for bill_cycle in range(len(unique_bc)):

        idx = bc_list == unique_bc[bill_cycle]

        disagg_sum = max(min_cons*Cgbdisagg.WH_IN_1_KWH, output_data[wh_idx, idx].sum())

        change_from_disagg = (disagg_sum - final_tou_consumption[wh_idx, idx].sum()) / disagg_sum * 100

        if (output_data[wh_idx, idx].sum() == 0) or (change_from_disagg > thres):
            continue

        # determines potential boxes to be removed

        change_from_disagg = final_tou_consumption[wh_idx, idx].sum() - (disagg_sum*(1-thres/100))

        potential_boxes = np.logical_and(final_tou_consumption[wh_idx] > 0, output_data[wh_idx] == 0)

        potential_boxes[np.logical_not(idx)] = 0

        factor = final_tou_consumption[wh_idx][potential_boxes].sum() / change_from_disagg

        if not (np.isnan(factor) or np.sum(potential_boxes) == 0):

            # removes new added boxes to maintain max 50% delta

            potential_boxes = potential_boxes.flatten()
            pot_box_seq = find_seq(potential_boxes > 0, np.zeros_like(potential_boxes), np.zeros_like(potential_boxes))
            pot_box_seq = pot_box_seq[pot_box_seq[:, 0] > 0]

            seed = RandomState(random_gen_config.seed_value)

            remove_wh_frac = int((1 / factor) * len(pot_box_seq))

            remove_wh_frac = min(remove_wh_frac, len(pot_box_seq))

            if remove_wh_frac <= 0:
                continue

            logger.debug("Reducing WH in billing cycle %s", bill_cycle)

            remove_boxes = seed.choice(np.arange(len(pot_box_seq)), remove_wh_frac, replace=False)

            wh_cons_copy = copy.deepcopy(final_tou_consumption[wh_idx])

            wh_cons_copy = wh_cons_copy.flatten()

            for k in range(len(remove_boxes)):
                wh_cons_copy[pot_box_seq[remove_boxes[k], 1] : pot_box_seq[remove_boxes[k], 2] + 1] = 0

            final_tou_consumption[wh_idx] = wh_cons_copy.reshape(final_tou_consumption[wh_idx].shape)

    final_tou_consumption = limit_wh_delta_for_inc(item_input_object, item_output_object, final_tou_consumption,
                                                   appliance_list, output_data, processed_input_data, hsm_wh, logger)

    final_tou_consumption = limit_wh_delta_for_mtd(item_input_object, item_output_object, final_tou_consumption,
                                                   appliance_list, output_data, processed_input_data, hsm_wh, logger)

    return final_tou_consumption


def block_low_cons_billing_cycle_output(item_input_object, final_tou_consumption, cooling_idx, heating_idx, pp_idx,
                                        ev_idx, ref_idx, wh_idx, li_idx, logger):
    """
    Block consumption in cases where appliance output is less than a certain limit

    Parameters:
        item_input_object          (dict)          : Dict containing all hybrid inputs
        final_tou_consumption      (np.ndarray)    : appliance ts level estimates (after adjustment)
        cooling_idx                (int)           : cooling index
        heating_idx                (int)           : heating index
        pp_idx                     (int)           : PP index
        ev_idx                     (int)           : EV index
        logger                     (logger)        : logger object

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    # blocking low consumption pp billing cycles

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    # blocking low consumption cooling/heating billing cycles
    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1) > 0

    cooling_limit = item_input_object.get('pilot_level_config').get('cooling_config').get('bounds').get('block_if_less_than') * 1000
    heating_limit = item_input_object.get('pilot_level_config').get('heating_config').get('bounds').get('block_if_less_than') * 1000
    pp_limit = item_input_object.get('pilot_level_config').get('pp_config').get('bounds').get('block_if_less_than') * 1000
    ev_limit = item_input_object.get('pilot_level_config').get('ev_config').get('bounds').get('block_if_less_than') * 1000

    wh_limit = item_input_object.get('pilot_level_config').get('wh_config').get('bounds').get('block_if_less_than') * 1000
    li_limit = item_input_object.get('pilot_level_config').get('li_config').get('bounds').get('block_if_less_than') * 1000
    ref_limit = item_input_object.get('pilot_level_config').get('ref_config').get('bounds').get('block_if_less_than') * 1000

    ev_limit = max(ev_limit, 5 * Cgbdisagg.WH_IN_1_KWH)
    pp_limit = max(pp_limit, 5 * Cgbdisagg.WH_IN_1_KWH)

    for i in range(len(unique_bc)):

        target_days_count = np.sum(bc_list == unique_bc[i])

        bc_days_count_scaling_factor = target_days_count / Cgbdisagg.DAYS_IN_MONTH

        if (np.sum(final_tou_consumption[pp_idx][bc_list == unique_bc[i]]) < (pp_limit * bc_days_count_scaling_factor)):
            logger.info("Blocking PP in BC | %s", int(unique_bc[i]))
            final_tou_consumption[pp_idx][bc_list == unique_bc[i]] = 0

        idx = cooling_idx
        vacation_count_scaling_factor = np.sum(vacation_days[bc_list == unique_bc[i]]) / target_days_count
        vacation_count_scaling_factor = 1 - vacation_count_scaling_factor

        if (np.sum(final_tou_consumption[idx][bc_list == unique_bc[i]]) <
                ((cooling_limit / Cgbdisagg.DAYS_IN_MONTH) * target_days_count)*vacation_count_scaling_factor):
            final_tou_consumption[idx][bc_list == unique_bc[i]] = 0

        idx = heating_idx

        if (np.sum(final_tou_consumption[idx][bc_list == unique_bc[i]]) <
                ((heating_limit / Cgbdisagg.DAYS_IN_MONTH) *  target_days_count) * vacation_count_scaling_factor):
            final_tou_consumption[idx][bc_list == unique_bc[i]] = 0

        idx = ev_idx

        if (np.sum(final_tou_consumption[idx][bc_list == unique_bc[i]]) <
                ((ev_limit / Cgbdisagg.DAYS_IN_MONTH) * target_days_count) * vacation_count_scaling_factor ):
            final_tou_consumption[idx][bc_list == unique_bc[i]] = 0

        idx = wh_idx

        if (np.sum(final_tou_consumption[idx][bc_list == unique_bc[i]]) <
                ((wh_limit / Cgbdisagg.DAYS_IN_MONTH) * target_days_count) * vacation_count_scaling_factor ):
            final_tou_consumption[idx][bc_list == unique_bc[i]] = 0

        idx = ref_idx

        if (np.sum(final_tou_consumption[idx][bc_list == unique_bc[i]]) <
                ((ref_limit / Cgbdisagg.DAYS_IN_MONTH) * target_days_count) * vacation_count_scaling_factor ):
            final_tou_consumption[idx][bc_list == unique_bc[i]] = 0

        idx = li_idx

        if (np.sum(final_tou_consumption[idx][bc_list == unique_bc[i]]) <
                ((li_limit / Cgbdisagg.DAYS_IN_MONTH) * target_days_count) * vacation_count_scaling_factor ):
            final_tou_consumption[idx][bc_list == unique_bc[i]] = 0

    return final_tou_consumption


def handle_leftover_neg_res_points(final_tou_consumption, ao_idx, samples_per_hour,
                                   processed_input_data):
    """
    Adjust consumption points if there is any overshoot of total itemization even after all adjustment

    Parameters:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
        ao_idx                      (int)           : always on appliance index
        samples_per_hour            (int)           : samples in an hour
        processed_input_data        (np.ndarray)    : input data

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    final_tou_consumption = np.fmax(0, final_tou_consumption)

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    if np.any(np.logical_or(-np.fmin(0, other_cons_arr) > (final_tou_consumption[ao_idx]-1),
                            other_cons_arr < -10 / samples_per_hour)):
        target_point = np.logical_or(-np.fmin(0, other_cons_arr) > (final_tou_consumption[ao_idx]-1),
                                     other_cons_arr < -10 / samples_per_hour)

        extra_cons = np.fmax(0, np.sum(np.nan_to_num(final_tou_consumption), axis=0) - processed_input_data)

        extra_cons[target_point == 0] = 0

        den = np.sum(np.nan_to_num(final_tou_consumption), axis=0)
        den[den == 0] = 1

        cons_frac = np.fmax(0, np.divide(final_tou_consumption, den))

        remove_cons = np.multiply(extra_cons, cons_frac)

        remove_cons = np.minimum(final_tou_consumption, remove_cons)
        remove_cons = np.nan_to_num(remove_cons)

        final_tou_consumption = final_tou_consumption - remove_cons

    return final_tou_consumption


def update_output_object_for_hld_change_cases(final_tou_consumption, item_input_object, output_data, wh_idx,
                                              pp_idx, ev_idx, logger):

    """
    This functions logs to scenarios of HLD change in hybrid v2

    Parameters:
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)
        item_input_object           (dict)          : Dict containing all hybrid inputs
        output_data                 (np.ndarray)    : disagg module output for all appliances
        wh_idx                      (int)           : wh index
        pp_idx                      (int)           : PP index
        ev_idx                      (int)           : EV index
        logger                      (logger)        : logger object

    Returns:
        item_input_object           (dict)          : Dict containing all hybrid inputs
    """

    item_input_object["item_input_params"]["wh_removed"] = 0
    item_input_object["item_input_params"]["wh_added"] = 0
    item_input_object["item_input_params"]["pp_added"] = 0
    item_input_object["item_input_params"]["ev_added"] = 0

    if np.sum(final_tou_consumption[wh_idx]) == 0 and np.sum(output_data[wh_idx]) > 0:
        item_input_object["item_input_params"]["wh_removed"] = 1
        logger.info('WH removed in itemization | ')

    if np.sum(final_tou_consumption[wh_idx]) > 0 and np.sum(output_data[wh_idx]) == 0:
        item_input_object["item_input_params"]["wh_added"] = 1
        logger.info('WH added in itemization | ')

    if np.sum(final_tou_consumption[ev_idx]) > 0 and np.sum(output_data[ev_idx]) == 0:
        item_input_object["item_input_params"]["ev_added"] = 1
        logger.info('EV added in itemization | ')

    if np.sum(final_tou_consumption[pp_idx]) > 0 and np.sum(output_data[pp_idx]) == 0:
        item_input_object["item_input_params"]["pp_added"] = 1
        logger.info('PP added in itemization | ')

    return item_input_object


def modify_stat_app_based_on_app_prof(final_tou_consumption, ld_idx, ent_idx, cook_idx, vacation):
    """
    Block statistical output for 0 app profile users

    Parameters:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
        ld_idx                      (int)           : laundry index
        ent_idx                     (int)           : entertainment index
        cook_idx                    (int)           : cooking index
        vacation                    (np.ndarray)    : vacation data
        logger                      (logger)        : logger object

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    final_tou_consumption[ld_idx, vacation, :] = 0
    final_tou_consumption[ent_idx, vacation, :] = 0
    final_tou_consumption[cook_idx, vacation, :] = 0

    # Final checks make sure that laundry is absent for 0 app profile user

    return final_tou_consumption


def apply_max_ts_level_limit(item_input_object, item_output_object, processed_input_data, appliance_list, output_data, final_tou_consumption):

    """
    This function ensure max ts level consumption. This function is added to avoid any high consumption
     being added after itemization

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object           (dict)         : Dict containing all hybrid outputs
        processed_input_data        (np.ndarray)    : raw input data
        appliance_list              (list)          : list of target appliances
        output_data                 (np.ndarray)    : disagg module output for all appliances
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    total_cons = processed_input_data[np.logical_not(vacation)]
    length = np.sum(np.logical_not(vacation))
    total_cons = ((np.sum(total_cons) / length) * (Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH))

    config = get_thresholds_config(total_cons).get('max_ts_lim')

    perc_for_max_wh_cap = config.get('perc_for_max_wh_cap')
    min_days_for_high_amp_wh = config.get('min_days_for_high_amp_wh')

    # Apply ts level limit

    pp_idx = np.where(appliance_list == 'pp')[0][0]
    cool_idx = np.where(appliance_list == 'cooling')[0][0]
    heat_idx = np.where(appliance_list == 'heating')[0][0]

    disagg_cool = output_data[cool_idx]
    disagg_heat = output_data[heat_idx]

    wh_limit = config.get('wh_limit')[1]/samples_per_hour

    if (len(final_tou_consumption[0]) < min_days_for_high_amp_wh) and (not (item_input_object.get("item_input_params").get("timed_wh_user"))):
        wh_limit = config.get('wh_limit')[0]/samples_per_hour

    wh_idx = np.where(appliance_list == 'wh')[0][0]
    disagg_wh = output_data[wh_idx]

    disagg_pp = output_data[pp_idx]

    if np.sum(disagg_wh > 0):
        wh_limit = 1.2 * np.percentile(disagg_wh[disagg_wh > 0], perc_for_max_wh_cap)

    if (item_input_object.get('config').get('disagg_mode') == 'mtd') and np.sum(disagg_wh > 0):
        wh_limit = np.percentile(disagg_wh[disagg_wh > 0], perc_for_max_wh_cap)

    # Apply ts level limit for hvac consumption

    final_tou_consumption[cool_idx] = np.fmin(max(np.max(disagg_cool), config.get('max_hvac_limit')/samples_per_hour), final_tou_consumption[cool_idx])
    final_tou_consumption[heat_idx] = np.fmin(max(np.max(disagg_heat), config.get('max_hvac_limit')/samples_per_hour), final_tou_consumption[heat_idx])
    final_tou_consumption[wh_idx][disagg_wh == 0] = np.fmin(wh_limit, final_tou_consumption[wh_idx][disagg_wh == 0])

    # Apply ts level limit for ev consumption

    ev_idx = np.where(appliance_list == 'ev')[0][0]
    disagg_ev = output_data[ev_idx]

    if np.any(disagg_ev > 0):
        final_tou_consumption[ev_idx][disagg_ev > 0] = np.fmin(disagg_ev[disagg_ev > 0]*config.get('max_ev_delta'),
                                                               final_tou_consumption[ev_idx][disagg_ev > 0])

    # Apply ts level limit for thin pulse in wh consumption

    if (item_input_object.get("item_input_params").get("final_thin_pulse") is not None) and (np.sum(final_tou_consumption[wh_idx]) > 0):
        thin_pulse = item_input_object.get("item_input_params").get("final_thin_pulse")
        thin_pulse_tou = thin_pulse > 0
        final_tou_consumption[np.where(appliance_list == 'wh')[0][0]][thin_pulse_tou] = \
            np.minimum(thin_pulse, final_tou_consumption[np.where(appliance_list == 'wh')[0][0]])[thin_pulse_tou]

    if (item_input_object.get('item_input_params').get('swh_hld') > 0) and disagg_wh.sum() > 0:
        final_tou_consumption[np.where(appliance_list == 'wh')[0][0]] = np.fmin(final_tou_consumption[np.where(appliance_list == 'wh')[0][0]], wh_limit)

    # Apply ts level limit for fat pulse in wh consumption

    if (item_input_object.get("item_input_params").get("final_fat_pulse") is not None) and \
            (np.sum(final_tou_consumption[wh_idx]) > 0) and np.sum(disagg_wh > 0):
        fat_pulse_tou = item_input_object.get("item_input_params").get("final_fat_pulse") > 0
        final_tou_consumption[np.where(appliance_list == 'wh')[0][0]][fat_pulse_tou] = \
            np.minimum(disagg_wh[fat_pulse_tou]*config.get('max_wh_active_pulse'),
                       final_tou_consumption[np.where(appliance_list == 'wh')[0][0]][fat_pulse_tou])

    if np.sum(final_tou_consumption[pp_idx]) > 0 and np.sum(disagg_pp) > 0:
        timed_output = item_output_object.get("timed_app_dict").get("pp")
        final_tou_consumption[pp_idx][timed_output == 0] = \
            np.minimum(disagg_pp[timed_output == 0], final_tou_consumption[pp_idx][timed_output == 0])

    if item_input_object["item_input_params"]["timed_wh_user"] > 0:
        final_tou_consumption[np.where(appliance_list == 'wh')[0][0]] = \
            np.fmin(final_tou_consumption[np.where(appliance_list == 'wh')[0][0]], config.get('wh_limit')[1]/samples_per_hour)

    if not item_input_object.get('item_input_params').get('run_hybrid_v2_flag'):
        return final_tou_consumption

    final_tou_consumption = apply_max_ts_level_limit_for_stat_app(item_input_object, item_output_object,
                                                                  processed_input_data, appliance_list,
                                                                  output_data, final_tou_consumption)

    return final_tou_consumption


def apply_max_ts_level_limit_for_stat_app(item_input_object, item_output_object, processed_input_data, appliance_list, output_data, final_tou_consumption):

    """
    This function ensure max ts level consumption. This function is added to avoid any high consumption
     being added after itemization

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object           (dict)         : Dict containing all hybrid outputs
        processed_input_data        (np.ndarray)    : raw input data
        appliance_list              (list)          : list of target appliances
        output_data                 (np.ndarray)    : disagg module output for all appliances
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    total_cons = processed_input_data[np.logical_not(vacation)]
    length = np.sum(np.logical_not(vacation))
    total_cons = ((np.sum(total_cons) / length) * (Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH))

    config = get_thresholds_config(total_cons).get('max_ts_lim')

    def_ld_ts_limit = config.get('def_ld_ts_limit')
    def_cook_ts_limit = config.get('def_cook_ts_limit')
    min_ent_cons_for_high_occ = config.get('min_ent_cons_for_high_occ')
    ent_cons_offset_based_on_occupancy = config.get('ent_cons_offset_based_on_occupancy')

    # Apply ts level limit

    ent_ts_limit = config.get('ent_ts_limit')

    if item_input_object.get('home_meta_data').get('numOccupants') is not None and \
            item_input_object.get('home_meta_data').get('numOccupants') >= 4:
        ent_ts_limit = max(min_ent_cons_for_high_occ[1], ent_ts_limit)
    if item_input_object.get('home_meta_data').get('numOccupants') is not None and \
            item_input_object.get('home_meta_data').get('numOccupants') >= 3:
        ent_ts_limit = max(min_ent_cons_for_high_occ[0], ent_ts_limit)

    office_goer_present = \
        item_output_object.get('occupants_profile').get('occupants_features')[item_output_object.get('occupants_profile').get('office_goer_index')]

    ent_ts_limit = ent_ts_limit - ent_cons_offset_based_on_occupancy * (bool(office_goer_present))

    stay_at_home_user_present = \
        item_output_object.get('occupants_profile').get('occupants_features')[item_output_object.get('occupants_profile').get('stay_at_home_index')]

    ent_ts_limit = ent_ts_limit + ent_cons_offset_based_on_occupancy * (bool(stay_at_home_user_present))

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    # Apply cooking/laundry ts level limit based on app profile information

    cook_ts_limit = get_max_ts_level_limit_for_cook(item_input_object, item_output_object, processed_input_data,
                                                    appliance_list, output_data, final_tou_consumption)

    ld_ts_limit = get_max_ts_level_limit_for_ld(item_input_object, item_output_object, processed_input_data)

    def_ent_ts_limit = ent_ts_limit

    if not item_input_object.get("appliance_profile").get("default_laundry_flag"):
        def_ld_ts_limit = ld_ts_limit
    if not item_input_object.get("appliance_profile").get("default_cooking_flag"):
        def_cook_ts_limit = cook_ts_limit

    ld_idx = hybrid_config.get("app_seq").index('ld')
    max_ts_level_lim = hybrid_config.get("max_ts_level_lim")[ld_idx]
    ts_level_lim_factor = hybrid_config.get("ts_level_lim_factor")[ld_idx]
    min_ts_level_lim = hybrid_config.get("min_ts_level_lim")[ld_idx]

    if max_ts_level_lim and item_input_object.get("appliance_profile").get("default_laundry_flag"):
        ld_ts_limit = min(ld_ts_limit, def_ld_ts_limit / ts_level_lim_factor)
    if min_ts_level_lim and item_input_object.get("appliance_profile").get("default_laundry_flag"):
        ld_ts_limit = max(ld_ts_limit, def_ld_ts_limit / ts_level_lim_factor)

    cook_idx = hybrid_config.get("app_seq").index('cook')
    max_ts_level_lim = hybrid_config.get("max_ts_level_lim")[cook_idx]
    ts_level_lim_factor = hybrid_config.get("ts_level_lim_factor")[cook_idx]
    min_ts_level_lim = hybrid_config.get("min_ts_level_lim")[cook_idx]

    if max_ts_level_lim and item_input_object.get("appliance_profile").get("default_cooking_flag"):
        cook_ts_limit = min(cook_ts_limit, def_cook_ts_limit / ts_level_lim_factor)
    if min_ts_level_lim and item_input_object.get("appliance_profile").get("default_cooking_flag"):
        cook_ts_limit = max(cook_ts_limit, def_cook_ts_limit / ts_level_lim_factor)

    ent_idx = hybrid_config.get("app_seq").index('ent')
    max_ts_level_lim = hybrid_config.get("max_ts_level_lim")[ent_idx]
    ts_level_lim_factor = hybrid_config.get("ts_level_lim_factor")[ent_idx]
    min_ts_level_lim = hybrid_config.get("min_ts_level_lim")[ent_idx]

    if max_ts_level_lim:
        ent_ts_limit = min(ent_ts_limit, def_ent_ts_limit / ts_level_lim_factor)
    if min_ts_level_lim:
        ent_ts_limit = max(ent_ts_limit, def_ent_ts_limit / ts_level_lim_factor)

    # Apply ts level limit for cooking/laundry/entertainment

    ent_idx = np.where(appliance_list == 'ent')[0][0]
    cook_idx = np.where(appliance_list == 'cook')[0][0]
    ld_idx = np.where(appliance_list == 'ld')[0][0]

    ld_ts_limit = max(0, ld_ts_limit)
    cook_ts_limit = max(0, cook_ts_limit)
    ent_ts_limit = max(0, ent_ts_limit)

    final_tou_consumption[ent_idx] = np.fmin(ent_ts_limit/samples_per_hour, final_tou_consumption[ent_idx])
    final_tou_consumption[cook_idx] = np.fmin(cook_ts_limit/samples_per_hour, final_tou_consumption[cook_idx])
    final_tou_consumption[ld_idx] = np.fmin(ld_ts_limit/samples_per_hour, final_tou_consumption[ld_idx])

    return final_tou_consumption


def get_max_ts_level_limit_for_cook(item_input_object, item_output_object, processed_input_data, appliance_list, output_data, final_tou_consumption):

    """
    This function ensure max ts level consumption. This function is added to avoid any high consumption
     being added after itemization

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object           (dict)         : Dict containing all hybrid outputs
        processed_input_data        (np.ndarray)    : raw input data
        appliance_list              (list)          : list of target appliances
        output_data                 (np.ndarray)    : disagg module output for all appliances
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    total_cons = processed_input_data[np.logical_not(vacation)]
    length = np.sum(np.logical_not(vacation))
    total_cons = ((np.sum(total_cons) / length) * (Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH))

    config = get_thresholds_config(total_cons).get('max_ts_lim')

    cook_default_app_cons = config.get('cook_default_app_cons')
    cook_amp_offset = config.get('cook_amp_offset')
    min_cook_amp = config.get('min_cook_amp')
    min_cook_amp_for_missing_app = config.get('min_cook_amp_for_missing_app')

    # Apply ts level limit

    cook_ts_limit = config.get('cook_ts_limit')
    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    # Apply cooking/laundry ts level limit based on app profile information

    if not item_input_object.get("appliance_profile").get("default_cooking_flag"):
        cooking_app_count = item_input_object.get("appliance_profile").get("cooking")
        cooking_app_type = item_input_object.get("appliance_profile").get("cooking_type")
        appliance_consumption = cook_default_app_cons

        ld_idx = hybrid_config.get("app_seq").index('cook')
        scale_app_cons = hybrid_config.get("scale_app_cons")[ld_idx]
        app_cons_factor = hybrid_config.get("scale_app_cons_factor")[ld_idx]

        if scale_app_cons:
            appliance_consumption = appliance_consumption * app_cons_factor

        cooking_app_count[cooking_app_type == 0] = cooking_app_count[cooking_app_type == 0] * 0
        appliance_consumption = np.dot(appliance_consumption, cooking_app_count)

        cook_ts_limit = appliance_consumption + cook_amp_offset

        if (hybrid_config.get("dishwash_cat") == "cook") and np.any(cooking_app_type == 0):
            cook_ts_limit = max(cook_ts_limit, min_cook_amp_for_missing_app)

        elif hybrid_config.get("dishwash_cat") == "cook":
            cook_ts_limit = max(cook_ts_limit, min_cook_amp)

    return cook_ts_limit


def get_max_ts_level_limit_for_ld(item_input_object, item_output_object, processed_input_data):

    """
    This function ensure max ts level consumption. This function is added to avoid any high consumption
     being added after itemization

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object           (dict)         : Dict containing all hybrid outputs
        processed_input_data        (np.ndarray)    : raw input data
    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """
    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    total_cons = processed_input_data[np.logical_not(vacation)]
    length = np.sum(np.logical_not(vacation))
    total_cons = ((np.sum(total_cons) / length) * (Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH))

    config = get_thresholds_config(total_cons).get('max_ts_lim')

    min_ld_amp = config.get('min_ld_amp')
    ld_amp_drop_for_missing_app = config.get('ld_amp_drop_for_missing_app')
    ld_default_app_cons = config.get('ld_default_app_cons')
    additional_amp_for_drier = config.get('additional_amp_for_drier')
    min_ld_amp_for_japan = config.get('min_ld_amp_for_japan')
    min_ld_amp_for_app_prof = config.get('min_ld_amp_for_app_prof')
    min_drier_amp = config.get('min_drier_amp')

    # Apply ts level limit
    ld_ts_limit = config.get('ld_ts_limit')

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    pilot_level_config = item_input_object.get('pilot_level_config')

    dishwasher_in_drop_app_list = ("31" in pilot_level_config.get('ld_config').get("drop_app")) or \
                                  ("33" in pilot_level_config.get('ld_config').get("drop_app"))

    dishwasher_in_drop_app_list_or_cook_cat = dishwasher_in_drop_app_list or (hybrid_config.get("dishwash_cat") == "cook")

    if not item_input_object.get("appliance_profile").get("default_laundry_flag"):
        laundry_app_count = item_input_object.get("appliance_profile").get("laundry")
        laundry_app_type = item_input_object.get("appliance_profile").get("laundry_type")
        appliance_consumption = np.array(ld_default_app_cons)

        ld_idx = hybrid_config.get("app_seq").index('ld')
        scale_app_cons = hybrid_config.get("scale_app_cons")[ld_idx]
        app_cons_factor = hybrid_config.get("scale_app_cons_factor")[ld_idx]

        pilot = item_input_object.get("config").get("pilot_id")

        if scale_app_cons:
            appliance_consumption = appliance_consumption * app_cons_factor
            appliance_consumption[2] = np.fmax(min_drier_amp, appliance_consumption[2])

            min_ld_amp_for_app_prof = min_ld_amp_for_app_prof * (pilot not in PilotConstants.HVAC_JAPAN_PILOTS) + \
                                      min_ld_amp_for_japan * (pilot in PilotConstants.HVAC_JAPAN_PILOTS)

            appliance_consumption[0] = np.fmax(min_ld_amp_for_app_prof, appliance_consumption[0])
            appliance_consumption[1] = np.fmax(min_ld_amp_for_app_prof, appliance_consumption[1])

        if dishwasher_in_drop_app_list_or_cook_cat:
            appliance_consumption[1] = 0

        if "6" in pilot_level_config.get('ld_config').get("drop_app"):
            appliance_consumption[2] = 0

        laundry_app_count[laundry_app_type == 0] = laundry_app_count[laundry_app_type == 0] * 0
        appliance_consumption = np.dot(appliance_consumption, laundry_app_count)

        ld_ts_limit = appliance_consumption

        if laundry_app_count[2]:
            ld_ts_limit = appliance_consumption + additional_amp_for_drier

    else:
        if hybrid_config.get("dishwash_cat") == "cook":
            ld_ts_limit = ld_ts_limit - ld_amp_drop_for_missing_app

        if "6" in pilot_level_config.get('ld_config').get("drop_app"):
            ld_ts_limit = ld_ts_limit - ld_amp_drop_for_missing_app

        if dishwasher_in_drop_app_list:
            ld_ts_limit = ld_ts_limit - ld_amp_drop_for_missing_app

        ld_ts_limit = max(min_ld_amp, ld_ts_limit)

    return ld_ts_limit


def limit_wh_delta_for_mtd(item_input_object, item_output_object, final_tou_consumption, appliance_list,
                           output_data, processed_input_data, hsm_wh, logger):

    """
    This function is added to limit BC level delta from historical to MTD mode
     (In order to control WH overestimation in hybrid module)

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)
        appliance_list              (list)          : list of appliances
        output_data                 (np.ndarray)    : ts level true disagg output for all appliances
        processed_input_data        (np.ndarray)    : ts level input data
        hsm_wh                      (int)           : wh monthly output in previous run

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    # remove new added boxes if change of wh from hist to mtd is more than 60%

    config = init_final_item_conf().get('post_processing_config')

    wh_idx = np.where(appliance_list == 'wh')[0][0]

    thres = -config.get('max_mtd_wh_delta')

    valid_idx = np.zeros(processed_input_data.shape)
    valid_idx[:, :] = 1
    valid_idx = valid_idx.astype(bool)

    delta_from_hsm = (hsm_wh - final_tou_consumption[wh_idx, valid_idx].sum()) / hsm_wh * 100

    timed_or_swh_user = \
        (not int(item_input_object.get("item_input_params").get("timed_wh_user"))) and \
        (not item_input_object.get('item_input_params').get('swh_hld'))

    if (item_input_object.get('config').get('disagg_mode') == 'mtd') and timed_or_swh_user and (delta_from_hsm <= thres) and (hsm_wh != 0):

        hsm_wh = hsm_wh * 1000 * 1.5

        logger.info("Reducing MTD WH output since delta from hsm is | %s", delta_from_hsm)

        delta_from_hsm = final_tou_consumption[wh_idx, valid_idx].sum() - (hsm_wh*(1-thres/100))

        potential_boxes = np.logical_and(final_tou_consumption[wh_idx] > 0, output_data[wh_idx] == 0)

        potential_boxes[np.logical_not(valid_idx)] = 0

        factor = final_tou_consumption[wh_idx][potential_boxes].sum() / delta_from_hsm

        # remove wh boxes to reduce wh in mtd mode

        if factor < 1:
            return final_tou_consumption

        potential_boxes = potential_boxes.flatten()
        pot_box_seq = find_seq(potential_boxes > 0, np.zeros_like(potential_boxes), np.zeros_like(potential_boxes))
        pot_box_seq = pot_box_seq[pot_box_seq[:, 0] > 0]

        seed = RandomState(random_gen_config.seed_value)

        frac = int((1/factor) * len(pot_box_seq))

        if frac <= 0:
            return final_tou_consumption

        remove_boxes = seed.choice(np.arange(len(pot_box_seq)), frac, replace=False)

        wh_cons_copy = copy.deepcopy(final_tou_consumption[wh_idx])

        wh_cons_copy = wh_cons_copy.flatten()

        for k in range(len(remove_boxes)):
            wh_cons_copy[pot_box_seq[remove_boxes[k], 1]: pot_box_seq[remove_boxes[k], 2] + 1] = 0

        final_tou_consumption[wh_idx] = wh_cons_copy.reshape(final_tou_consumption[wh_idx].shape)

    return final_tou_consumption


def limit_wh_delta_for_inc(item_input_object, item_output_object, final_tou_consumption, appliance_list,
                           output_data, processed_input_data, hsm_wh, logger):

    """
    This function is added to limit BC level delta from historical to MTD mode
     (In order to control WH overestimation in hybrid module)

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)
        appliance_list              (list)          : list of appliances
        output_data                 (np.ndarray)    : ts level true disagg output for all appliances
        processed_input_data        (np.ndarray)    : ts level input data
        hsm_wh                      (int)           : wh monthly output in previous run

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    # remove new added boxes if change of wh from hist to mtd is more than 60%

    config = init_final_item_conf().get('post_processing_config')

    wh_idx = np.where(appliance_list == 'wh')[0][0]

    thres = -config.get('max_mtd_wh_delta')

    valid_idx = np.zeros(processed_input_data.shape)
    valid_idx[:, :] = 1
    valid_idx = valid_idx.astype(bool)

    delta_from_hsm = (hsm_wh - final_tou_consumption[wh_idx, valid_idx].sum()) / hsm_wh * 100

    timed_or_swh_user = \
        (not int(item_input_object.get("item_input_params").get("timed_wh_user"))) and \
        (not item_input_object.get('item_input_params').get('swh_hld'))

    if hsm_wh == 0 or np.sum(final_tou_consumption[wh_idx]) == 0:
        return final_tou_consumption

    user_has_storage_wh = \
        (item_input_object.get('config').get('disagg_mode') == 'incremental') and \
        (timed_or_swh_user) and (output_data[wh_idx].sum() == 0)

    hsm_wh = hsm_wh * 1000 * 1.2

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]

    output_billing_cycles = item_input_object.get('out_bill_cycles_by_module').get('disagg_bc')[:, 0]

    if np.any(np.isin(bc_list, output_billing_cycles)):
        valid_idx = np.isin(bc_list, output_billing_cycles)

    if (np.sum(valid_idx) == 0) or (not user_has_storage_wh):
        return final_tou_consumption

    logger.info("Reducing MTD WH output since delta from hsm is | %s", delta_from_hsm)

    delta_from_hsm = final_tou_consumption[wh_idx, valid_idx].sum() - (hsm_wh*(1-thres/100))

    potential_boxes = np.logical_and(final_tou_consumption[wh_idx, valid_idx] > 0, output_data[wh_idx, valid_idx] == 0)

    scaling_fact_based_on_diff_with_hsm = final_tou_consumption[wh_idx, valid_idx][potential_boxes].sum() / delta_from_hsm

    # remove wh boxes to reduce additional wh in inc mode

    potential_boxes = potential_boxes.flatten()
    pot_box_seq = find_seq(potential_boxes > 0, np.zeros_like(potential_boxes), np.zeros_like(potential_boxes))
    pot_box_seq = pot_box_seq[pot_box_seq[:, 0] > 0]

    seed = RandomState(random_gen_config.seed_value)

    temp_cons = copy.deepcopy(final_tou_consumption[wh_idx])

    frac = int(np.nan_to_num(1/scaling_fact_based_on_diff_with_hsm) * len(pot_box_seq))

    if frac > 0 or scaling_fact_based_on_diff_with_hsm >= 1:
        remove_boxes = seed.choice(np.arange(len(pot_box_seq)), frac, replace=False)

        wh_cons_copy = copy.deepcopy(final_tou_consumption[wh_idx, valid_idx])

        wh_cons_copy = wh_cons_copy.flatten()

        for k in range(len(remove_boxes)):
            wh_cons_copy[pot_box_seq[remove_boxes[k], 1]: pot_box_seq[remove_boxes[k], 2] + 1] = 0

        temp_cons[valid_idx] = wh_cons_copy.reshape(final_tou_consumption[wh_idx, valid_idx].shape)

        final_tou_consumption[wh_idx] = temp_cons

    return final_tou_consumption


def get_stat_app_hsm(item_input_object, processed_input_data, cook_cons_based_on_hsm, ent_cons_based_on_hsm, ld_cons_based_on_hsm):

    """
    Fetch and process hsm data of cooking/laundry/entertainment category

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        processed_input_data        (np.ndarray)    : raw input data
        cook_cons_based_on_hsm      (float)         : cooking consumption
        ent_cons_based_on_hsm       (float)         : ent consumption
        ld_cons_based_on_hsm        (float)         : ld consumption

    Returns:
        use_hsm                     (bool)          : flag to represent whether hsm data is valid
        cook_cons_based_on_hsm      (float)         : cooking consumption updated using hsm data
        ent_cons_based_on_hsm       (float)         : ent consumption updated using hsm data
        ld_cons_based_on_hsm        (float)         : ld consumption updated using hsm data
    """

    use_hsm = 0

    valid_stat_hsm = item_input_object.get("item_input_params").get('valid_life_hsm')

    hsm_weights_in_mtd = 0.9
    hsm_weights_in_inc = [0.5, 0.75, 0.9]
    valid_stat_app_hsm_present = \
        valid_stat_hsm and item_input_object.get("item_input_params").get('life_hsm') is not None and \
        item_input_object.get("item_input_params").get('life_hsm').get('ent_cons') is not None

    if not valid_stat_app_hsm_present:
        return use_hsm, cook_cons_based_on_hsm, ent_cons_based_on_hsm, ld_cons_based_on_hsm

    ent_hsm_input = item_input_object.get("item_input_params").get('life_hsm').get('ent_cons')

    if ent_hsm_input is not None and isinstance(ent_hsm_input, list):
        ent_hsm_input = ent_hsm_input[0]

    cook_hsm_input = item_input_object.get("item_input_params").get('life_hsm').get('cook_cons')

    if cook_hsm_input is not None and isinstance(cook_hsm_input, list):
        cook_hsm_input = cook_hsm_input[0]

    ld_hsm_input = item_input_object.get("item_input_params").get('life_hsm').get('ld_cons')

    if ld_hsm_input is not None and isinstance(ld_hsm_input, list):
        ld_hsm_input = ld_hsm_input[0]

    if item_input_object.get('config').get('disagg_mode') == 'mtd':
        ent_cons_based_on_hsm = ent_hsm_input * hsm_weights_in_mtd + ent_cons_based_on_hsm * (1-hsm_weights_in_mtd)
        cook_cons_based_on_hsm = cook_hsm_input * hsm_weights_in_mtd + cook_cons_based_on_hsm * (1-hsm_weights_in_mtd)
        ld_cons_based_on_hsm = ld_hsm_input * hsm_weights_in_mtd + ld_cons_based_on_hsm * (1-hsm_weights_in_mtd)
        use_hsm = 1

    if (item_input_object.get('config').get('disagg_mode') == 'incremental') and len(processed_input_data) < 40:
        ent_cons_based_on_hsm = ent_cons_based_on_hsm * (1-hsm_weights_in_inc[0]) + ent_hsm_input * (hsm_weights_in_inc[0])
        cook_cons_based_on_hsm = cook_cons_based_on_hsm * (1-hsm_weights_in_inc[0]) + cook_hsm_input * (hsm_weights_in_inc[0])
        ld_cons_based_on_hsm = ld_cons_based_on_hsm * (1-hsm_weights_in_inc[0]) + ld_hsm_input * (hsm_weights_in_inc[0])
        use_hsm = 1

    elif (item_input_object.get('config').get('disagg_mode') == 'incremental') and len(processed_input_data) < 70:
        ent_cons_based_on_hsm = ent_cons_based_on_hsm * (1-hsm_weights_in_inc[1]) + ent_hsm_input * (hsm_weights_in_inc[1])
        cook_cons_based_on_hsm = cook_cons_based_on_hsm * (1-hsm_weights_in_inc[1]) + cook_hsm_input * (hsm_weights_in_inc[1])
        ld_cons_based_on_hsm = ld_cons_based_on_hsm * (1-hsm_weights_in_inc[1]) + ld_hsm_input * (hsm_weights_in_inc[1])
        use_hsm = 1

    elif (item_input_object.get('config').get('disagg_mode') == 'incremental') and len(processed_input_data) >= 100:
        ent_cons_based_on_hsm = ent_hsm_input *(hsm_weights_in_inc[2]) + ent_cons_based_on_hsm * (1-hsm_weights_in_inc[2])
        cook_cons_based_on_hsm = cook_hsm_input * (hsm_weights_in_inc[2]) + cook_cons_based_on_hsm * (1-hsm_weights_in_inc[2])
        ld_cons_based_on_hsm = ld_hsm_input * (hsm_weights_in_inc[2]) + ld_cons_based_on_hsm * (1-hsm_weights_in_inc[2])
        use_hsm = 1

    return use_hsm, cook_cons_based_on_hsm, ent_cons_based_on_hsm, ld_cons_based_on_hsm
