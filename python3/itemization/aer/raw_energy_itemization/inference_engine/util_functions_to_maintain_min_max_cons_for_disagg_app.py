

"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Prepare data required in itemization pipeline
"""

# Import python packages

import copy
import numpy as np
import pandas as pd

from numpy.random import RandomState

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.functions.get_config import get_hybrid_config

from python3.itemization.init_itemization_config import random_gen_config

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import get_index

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.hsm_utils import get_backup_app_hsm


def maintian_wh_max_cons(appliance_list, input_data, final_tou_consumption, item_input_object, item_output_object, unique_bc, bc_list, logger):

    """
    This function maintains a max cap on billing cycle level wh consumption

    Parameters:
        appliance_list            (np.ndarray)    : list of all appliances
        input_data                (np.ndarray)    :  user input data
        final_tou_consumption     (np.ndarray)    : hybrid v2 output for all appliances
        item_input_object         (dict)          : dict containing all input
        item_output_object        (dict)          : dict containing all output
        unique_bc                 (np.ndarray)    : list of unique billing cycles
        bc_list                   (np.ndarray)    : billing cycle data
        logger                    (np.ndarray)    : logger object

    Returns:
        final_tou_consumption     (np.ndarray)    : hybrid v2 output for all appliances
    """

    # preparing vacation and billing cycle data

    max_cons = item_input_object.get('pilot_level_config').get('wh_config').get('bounds').get('max_cons')

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    wh_idx = np.where(appliance_list == 'wh')[0][0]

    wh_cons = final_tou_consumption[wh_idx]

    # not adding max cap if wh is added from step3 addition

    if 'wh' in item_input_object["item_input_params"]["backup_app"]:
        return final_tou_consumption

    for bc in unique_bc:

        target_days = bc_list == bc

        vac_factor = np.sum(vacation[target_days]) / np.sum(target_days)
        vac_factor = 1 - vac_factor

        # checking the extra wh cons

        diff = final_tou_consumption[wh_idx][target_days].sum() / 1000 - max_cons * vac_factor * (np.sum(target_days) / 30)

        if diff > 0 and ('wh' not in item_input_object["item_input_params"]["backup_app"]) and np.sum(wh_cons[target_days]) > 0:

            potential_boxes = wh_cons[target_days]

            potential_boxes_1d = wh_cons[target_days].flatten()

            if np.sum(potential_boxes) == 0:
                continue

            factor = diff / (potential_boxes.sum() / 1000)

            # removing extra wh box consumption

            potential_boxes = potential_boxes.flatten()
            pot_box_seq = find_seq(potential_boxes > 0, np.zeros_like(potential_boxes), np.zeros_like(potential_boxes))
            pot_box_seq = pot_box_seq[pot_box_seq[:, 0] > 0]

            seed = RandomState(random_gen_config.seed_value)

            remove_wh_frac = int((factor) * len(pot_box_seq))
            remove_wh_frac = min(remove_wh_frac, len(pot_box_seq))

            remove_boxes = seed.choice(np.arange(len(pot_box_seq)), remove_wh_frac, replace=False)

            for k in range(len(remove_boxes)):
                potential_boxes_1d[pot_box_seq[remove_boxes[k], 1]: pot_box_seq[remove_boxes[k], 2] + 1] = 0

            wh_cons[target_days] = potential_boxes_1d.reshape(wh_cons[target_days].shape)

    final_tou_consumption[wh_idx] = wh_cons

    return final_tou_consumption


def maintain_wh_min_cons(appliance_list, input_data, final_tou_consumption, item_input_object, item_output_object, unique_bc, bc_list, logger):
    """
    This function maintains a min cap on billing cycle level wh consumption

    Parameters:
        appliance_list            (np.ndarray)    : list of all appliances
        input_data                (np.ndarray)    :  user input data
        final_tou_consumption     (np.ndarray)    : hybrid v2 output for all appliances
        item_input_object         (dict)          : dict containing all input
        item_output_object        (dict)          : dict containing all output
        unique_bc                 (np.ndarray)    : list of unique billing cycles
        bc_list                   (np.ndarray)    : billing cycle data
        logger                    (np.ndarray)    : logger object

    Returns:
        final_tou_consumption     (np.ndarray)    : hybrid v2 output for all appliances
    """

    # preparing vacation and billing cycle data

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    min_cons = item_input_object.get('pilot_level_config').get('wh_config').get('bounds').get('min_cons')

    other_cons_arr = np.fmax(0, input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0))
    wh_idx = np.where(appliance_list == 'wh')[0][0]

    # not checking min cons if wh is added from step 3 addition or detection is 0

    if ('wh' in item_input_object["item_input_params"]["backup_app"]) or (np.sum(final_tou_consumption[wh_idx] > 0) == 0):
        return final_tou_consumption

    non_wh_hours = np.sum(final_tou_consumption[wh_idx], axis=0) == 0

    samples = int(final_tou_consumption.shape[2] /  Cgbdisagg.HRS_IN_DAY)

    inactive_hours = np.arange(2*samples, 5*samples + 1)

    median_val = np.median(final_tou_consumption[wh_idx][final_tou_consumption[wh_idx] > 0])

    wh_cons = final_tou_consumption[wh_idx]

    if median_val <= 0:
        return final_tou_consumption

    for bc in unique_bc:

        vac_factor = np.sum(vacation[bc_list == bc]) / np.sum(bc_list == bc)
        vac_factor = 1 - vac_factor

        diff = min_cons * vac_factor * (np.sum(bc_list == bc) / 30) - final_tou_consumption[wh_idx][bc_list == bc].sum() / 1000

        if not (diff > 0 and np.sum(wh_cons[bc_list == bc]) > 0):
            continue

        additional_cons = copy.deepcopy(other_cons_arr[bc_list == bc])

        additional_cons[vacation[bc_list == bc]] = 0

        additional_cons[:, inactive_hours] = 0

        additional_cons[additional_cons < 0.9 * median_val] = 0
        additional_cons[:, non_wh_hours] = 0

        additional_cons = np.fmin(additional_cons, 1.1 * median_val).flatten()
        additional_cons[additional_cons > 0] = median_val

        if ((final_tou_consumption[wh_idx][bc_list == bc].sum() < 4000) or
                (final_tou_consumption[wh_idx][bc_list == bc] > 0).sum() < 5 * samples):
            continue

        temp_seq = find_seq(additional_cons > 0, np.zeros_like(additional_cons), np.zeros_like(additional_cons), overnight=0)

        for j in range(len(temp_seq)):
            if temp_seq[j, 0] > 0 and temp_seq[j, 3] > 4 * samples:
                additional_cons[temp_seq[j, 1]: temp_seq[j, 2] + 1] = 0

        if np.sum(additional_cons) == 0:
            continue

        factor = 1 - min(0.99, (diff / (additional_cons.sum() / 1000)))

        pot_box_seq = np.where(additional_cons > 0)[0]

        seed = RandomState(random_gen_config.seed_value)

        remove_wh_frac = int((factor) * len(pot_box_seq))
        remove_wh_frac = min(remove_wh_frac, len(pot_box_seq))

        remove_boxes = seed.choice(np.arange(len(pot_box_seq)), remove_wh_frac, replace=False)

        additional_cons[pot_box_seq[remove_boxes]] = 0

        wh_cons[bc_list == bc] = wh_cons[bc_list == bc] + additional_cons.reshape(wh_cons[bc_list == bc].shape)

    final_tou_consumption[wh_idx] = wh_cons

    return final_tou_consumption


def maintain_min_cons_for_step3_app(appliance_list, input_data, final_tou_consumption, item_input_object,
                                    item_output_object, unique_bc, bc_list, logger):
    """
    This function maintains a min cap on billing cycle level for step3 appliances

    Parameters:
        appliance_list            (np.ndarray)    : list of all appliances
        input_data                (np.ndarray)    :  user input data
        final_tou_consumption     (np.ndarray)    : hybrid v2 output for all appliances
        item_input_object         (dict)          : dict containing all input
        item_output_object        (dict)          : dict containing all output
        unique_bc                 (np.ndarray)    : list of unique billing cycles
        bc_list                   (np.ndarray)    : billing cycle data
        logger                    (np.ndarray)    : logger object

    Returns:
        final_tou_consumption     (np.ndarray)    : hybrid v2 output for all appliances
    """

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    # lopping over step 3 appliances

    use_hsm, hsm_output = get_backup_app_hsm(item_input_object, item_input_object["item_input_params"]["backup_app"], logger)

    for app in item_input_object["item_input_params"]["backup_app"]:

        other_cons_arr = input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

        max_limit = item_input_object.get('pilot_level_config').get(app + '_config').get('bounds').get('min_cons')

        app_idx = np.where(appliance_list == app)[0][0]

        for bc in unique_bc:

            target_days = bc_list == bc

            # calculating vacation based scaling factor

            vac_factor = np.sum(vacation[target_days]) / np.sum(target_days)
            vac_factor = 1 - vac_factor

            vac_factor = 1 * (app == 'pp') + vac_factor * (app != 'pp')

            app_cons = final_tou_consumption[app_idx, target_days, :][np.logical_not(vacation[target_days])]
            app_cons = ((np.sum(app_cons) / len(final_tou_consumption[app_idx, target_days, :])) * (30 / 1000))

            if not ((1 - (np.sum(vacation[target_days]) / np.sum(target_days))) == 0):
                app_cons = app_cons / (1 - (np.sum(vacation[target_days]) / np.sum(target_days)))

            j = np.where(np.array(item_input_object["item_input_params"]["backup_app"]) == app)[0][0]
            if use_hsm[j]:
                max_limit = max(max_limit, hsm_output[j] * 0.9)

            if (app_cons < max_limit*vac_factor):

                # if consumption is less than limit,
                # checking if extra consumption is available in residual data

                score = np.zeros_like(input_data[target_days])

                app_tou = np.ones_like(input_data[0]).astype(bool)

                other_cons_arr = np.fmax(0, other_cons_arr)
                score[:, app_tou] = np.fmin(10000, other_cons_arr[target_days][:, app_tou]) / (np.max(np.fmin(10000, other_cons_arr[target_days][:, app_tou])))
                score = np.nan_to_num(score)

                samples = int(input_data.shape[1]/Cgbdisagg.HRS_IN_DAY)
                score[input_data[target_days] < 500/samples] = 0

                if np.sum(np.nan_to_num(score)) > 0:

                    # picking extra consumption at ts level from residual data , where total equals to required cons

                    score = score / np.sum(score)

                    score = np.nan_to_num(score)

                    score[vacation[target_days] > 0] = 0

                    diff = (max_limit*vac_factor - app_cons) * 1000 * (np.sum(target_days) / Cgbdisagg.DAYS_IN_MONTH)

                    score = score * diff

                    other_cons_arr = input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

                    # adding extra cons into step 3 appliance

                    final_tou_consumption[app_idx, target_days, :] = final_tou_consumption[app_idx, target_days, :] + \
                                                                     np.fmax(0, np.minimum(other_cons_arr[target_days, :] * 0.99, score).astype(int))

                    final_tou_consumption[app_idx, target_days, :] = np.minimum(final_tou_consumption[app_idx, target_days, :], input_data[target_days])

            other_cons_arr = np.fmax(0, input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0))

    return final_tou_consumption


def apply_max_cap_on_ref(appliance_list, input_data, final_tou_consumption, item_input_object,
                         item_output_object, unique_bc, bc_list, logger):
    """
    This function maintains a max cap on billing cycle level for ref appliance

    Parameters:
        appliance_list            (np.ndarray)    : list of all appliances
        input_data                (np.ndarray)    :  user input data
        final_tou_consumption     (np.ndarray)    : hybrid v2 output for all appliances
        item_input_object         (dict)          : dict containing all input
        item_output_object        (dict)          : dict containing all output
        unique_bc                 (np.ndarray)    : list of unique billing cycles
        bc_list                   (np.ndarray)    : billing cycle data
        logger                    (np.ndarray)    : logger object

    Returns:
        final_tou_consumption     (np.ndarray)    : hybrid v2 output for all appliances
    """

    max_limit = item_input_object.get('pilot_level_config').get('ref_config').get('bounds').get('max_cons')

    missing_days = np.sum(input_data > 0, axis=1) == 0

    for i, app in enumerate(['ref']):

        for bc in unique_bc:

            # reducing ref consumption if it is greater than a certain limit

            target_days = bc_list == bc

            season_val = 1

            vac_factor = np.sum(missing_days[target_days]) / np.sum(target_days)
            vac_factor = 1 - vac_factor

            app_idx = np.where(appliance_list == app)[0][0]

            factor = (final_tou_consumption[app_idx][target_days].sum() / (max_limit * 1000 * vac_factor* season_val)) * \
                     Cgbdisagg.DAYS_IN_MONTH / np.sum(target_days)

            if factor > 1:
                final_tou_consumption[app_idx][target_days] = final_tou_consumption[app_idx][target_days] / factor

    return final_tou_consumption


def maintain_min_cons_for_disagg_app(appliance_list, input_data, final_tou_consumption, item_input_object,
                                     item_output_object, unique_bc, bc_list, others_cons_arr):

    """
    This function maintains a min cap on billing cycle level for 'ref', 'li', 'cooling', 'heating'

    Parameters:
        appliance_list            (np.ndarray)    : list of all appliances
        input_data                (np.ndarray)    :  user input data
        final_tou_consumption     (np.ndarray)    : hybrid v2 output for all appliances
        item_input_object         (dict)          : dict containing all input
        item_output_object        (dict)          : dict containing all output
        unique_bc                 (np.ndarray)    : list of unique billing cycles
        bc_list                   (np.ndarray)    : billing cycle data
        logger                    (np.ndarray)    : logger object

    Returns:
        final_tou_consumption     (np.ndarray)    : hybrid v2 output for all appliances
    """

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    ref_season = hybrid_config.get('ref_season')

    date_list = item_output_object.get("date_list")
    month_list = pd.DatetimeIndex(date_list).month.values

    missing_days = np.sum(input_data > 0, axis=1) == 0

    for app in ['ref', 'li', 'cooling', 'heating']:

        max_limit = item_input_object.get('pilot_level_config').get(app + '_config').get('bounds').get('min_cons')

        for bc in unique_bc:

            target_days = bc_list == bc

            app_idx = np.where(appliance_list == app)[0][0]

            # preparing scaling factor based on vacation or missing days

            vac_factor = np.sum(vacation[target_days]) / np.sum(target_days)
            vac_factor = 1 - vac_factor

            season_val = ref_season[(month_list[target_days] - 1).astype(int)].mean() + 1

            if app in ['ref']:
                vac_factor = np.sum(missing_days[target_days]) / np.sum(target_days)
                vac_factor = 1 - vac_factor

            # reducing consumption is it is greater than required threshold

            factor = (final_tou_consumption[app_idx][target_days].sum() / (max_limit*season_val  * 1000*vac_factor)) * 30/np.sum(target_days)

            if factor < 1 and np.sum(final_tou_consumption[app_idx][target_days]) > 0:
                desired_val = final_tou_consumption[app_idx][target_days] / factor  - final_tou_consumption[app_idx][target_days]

                desired_val = np.minimum(desired_val, others_cons_arr[target_days])

                final_tou_consumption[app_idx][target_days] = final_tou_consumption[app_idx][target_days] + desired_val

        others_cons_arr = np.fmax(0, input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0))

    return final_tou_consumption


def maintain_minimum_thin_pulse_cons(min_cons_vals, mid_cons_vals, max_cons_vals, item_input_object, input_data,
                                     output_data, swh_pilots, logger):

    """
    this function makes sure that wh thin pulse output is present in itemization ts level output

    Parameters:
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        item_input_object         (dict)          : dict containing all input
        output_data               (np.ndarray)    : disagg ts level output output
        input_data                (np.ndarray)    : raw input data
        swh_pilots                (np.ndarray)    : list of swh pilots
        logger                    (logger)        : logger object
    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
    """

    wh_idx = get_index('wh')
    cooling_idx = get_index('cooling')
    heating_idx = get_index('heating')

    # no estimate change shud be added to thin pulses consumption of wh output

    thin_pulse = item_input_object.get("item_input_params").get("final_thin_pulse")
    pilot = item_input_object.get('config').get('pilot_id')

    if item_input_object.get('item_input_params').get('tankless_wh') > 0 and np.sum(output_data[wh_idx]) == 0 and (thin_pulse is not None):
        thin_pulse[:, :] = 0

    if thin_pulse is not None and np.sum(output_data[wh_idx]) > 0 and\
            (not item_input_object.get("item_input_params").get("timed_wh_user")) and (pilot not in swh_pilots):
        diff = input_data - thin_pulse
        logger.info("Checking minimum ts level thin pulse consumption | ")
        min_cons_vals[cooling_idx][thin_pulse > 0] = np.minimum(min_cons_vals[cooling_idx], diff)[thin_pulse > 0]
        min_cons_vals[heating_idx][thin_pulse > 0] = np.minimum(min_cons_vals[heating_idx], diff)[thin_pulse > 0]
        mid_cons_vals[cooling_idx][thin_pulse > 0] = np.minimum(mid_cons_vals[cooling_idx], diff)[thin_pulse > 0]
        mid_cons_vals[heating_idx][thin_pulse > 0] = np.minimum(mid_cons_vals[heating_idx], diff)[thin_pulse > 0]
        mid_cons_vals[wh_idx][thin_pulse > 0] = np.maximum(mid_cons_vals[wh_idx], thin_pulse)[thin_pulse > 0]
        max_cons_vals[wh_idx][thin_pulse > 0] = np.maximum(max_cons_vals[wh_idx], thin_pulse)[thin_pulse > 0]
        min_cons_vals[wh_idx][thin_pulse > 0] = np.maximum(min_cons_vals[wh_idx], thin_pulse)[thin_pulse > 0]

    return min_cons_vals, mid_cons_vals, max_cons_vals


def block_low_cons_pp(min_cons_vals, mid_cons_vals, max_cons_vals, logger, item_input_object, output_data):
    """
    blocking of pp for low consuption billing cycles and blocking of hvac based on app profile

    Parameters:
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        logger                    (logger)        : logger
        item_input_object         (dict)          : dict containing all input
        output_data               (np.ndarray)    : disagg ts level output output

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
    """

    pp_idx = get_index('pp')

    disagg_cons = output_data[pp_idx]

    # blocking low consumption pp billing cycles

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    for i in range(len(unique_bc)):

        scale = np.sum(bc_list == unique_bc[i]) / Cgbdisagg.DAYS_IN_MONTH

        low_cons_pp = (np.sum(disagg_cons[bc_list == unique_bc[i]]) < 10000 * scale) and \
                      (np.sum(max_cons_vals[pp_idx][bc_list == unique_bc[i]]) < 30000 * scale)

        if low_cons_pp:
            logger.info("Blocking PP in certain BC | %s", int(unique_bc[i]))
            mid_cons_vals[pp_idx][bc_list == unique_bc[i]] = 0
            max_cons_vals[pp_idx][bc_list == unique_bc[i]] = 0
            min_cons_vals[pp_idx][bc_list == unique_bc[i]] = 0

    return min_cons_vals, mid_cons_vals, max_cons_vals


def handle_low_cons(item_input_object, item_output_object, min_cons_vals, mid_cons_vals, max_cons_vals,
                    vacation, output_data, logger, input_data, swh_pilots):

    """
    blocking of pp for low consuption billing cycles and blocking of hvac based on app profile

    Parameters:
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        logger                    (logger)        : logger
        item_input_object         (dict)          : dict containing all input
        output_data               (np.ndarray)    : disagg ts level output output

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
    """

    wh_idx = get_index('wh')
    cooling_idx = get_index('cooling')

    # checking if AO/HVAC AO is unchanged after adjusting ranges
    min_cons_vals, mid_cons_vals, max_cons_vals = \
        maintain_min_ao_cons(min_cons_vals, mid_cons_vals, max_cons_vals, item_input_object, output_data, input_data)

    max_cons_vals[cooling_idx] = np.maximum(max_cons_vals[cooling_idx], output_data[cooling_idx])
    max_cons_vals[wh_idx] = np.maximum(max_cons_vals[wh_idx], mid_cons_vals[wh_idx])

    min_cons_vals[get_index('ref')] = np.maximum(mid_cons_vals[get_index('ref')], min_cons_vals[get_index('ref')])
    min_cons_vals[get_index('ao')] = np.maximum(mid_cons_vals[get_index('ao')], min_cons_vals[get_index('ao')])

    # block low bc level hvac consumption
    min_cons_vals, mid_cons_vals, max_cons_vals = \
        block_low_cons_hvac(item_input_object, item_output_object, output_data, min_cons_vals, mid_cons_vals, max_cons_vals, swh_pilots, logger)

    # postprocessing pp consumption to maintain ts level consistency

    ao_idx = get_index('ao')
    ref_idx = get_index('ref')

    min_cons_vals[ao_idx][vacation] = np.minimum(min_cons_vals[ao_idx], input_data-mid_cons_vals[ref_idx])[vacation]
    mid_cons_vals[ao_idx][vacation] = np.minimum(mid_cons_vals[ao_idx], input_data-mid_cons_vals[ref_idx])[vacation]
    max_cons_vals[ao_idx][vacation] = np.minimum(max_cons_vals[ao_idx], input_data-mid_cons_vals[ref_idx])[vacation]

    return min_cons_vals, mid_cons_vals, max_cons_vals


def maintain_min_ao_cons(min_cons_vals, mid_cons_vals, max_cons_vals, item_input_object, output_data, input_data):
    """
    this function makes sure that AO/ref is present in itemization ts level output

    Parameters:
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        item_input_object         (dict)          : Dict containing all hybrid inputs
        output_data               (np.ndarray)    : disagg ts level output output
        input_data                (np.ndarray)    : raw input data
    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
    """

    ent_idx = get_index('ent')
    cook_idx = get_index('cook')
    ld_idx = get_index('ld')

    pp_idx = get_index('pp')
    wh_idx = get_index('wh')
    ev_idx = get_index('ev')

    cooling_idx = get_index('cooling')
    heating_idx = get_index('heating')

    ao_cool = item_input_object.get("item_input_params").get("ao_cool")
    ao_heat = item_input_object.get("item_input_params").get("ao_heat")
    ao_output = output_data[get_index('ao')]
    ref_output = output_data[get_index('ref')]

    # maintain ao/ref  ts level output same as disagg output,
    # incase of overshoot of total disagg values, consumption is decreased on other appliances rather than ao/ref


    if np.sum(ao_output) > 0:
        diff = input_data - ao_output - ref_output
        min_cons_vals[cook_idx][ao_output > 0] = np.minimum(min_cons_vals[cook_idx], diff)[ao_output > 0]
        min_cons_vals[ld_idx][ao_output > 0] = np.minimum(min_cons_vals[ld_idx], diff)[ao_output > 0]
        min_cons_vals[ent_idx][ao_output > 0] = np.minimum(min_cons_vals[ent_idx], diff)[ao_output > 0]

        min_cons_vals[wh_idx][ao_output > 0] = np.minimum(min_cons_vals[wh_idx], diff)[ao_output > 0]
        min_cons_vals[pp_idx][ao_output > 0] = np.minimum(min_cons_vals[pp_idx], diff)[ao_output > 0]
        min_cons_vals[ev_idx][ao_output > 0] = np.minimum(min_cons_vals[ev_idx], diff)[ao_output > 0]
        min_cons_vals[cooling_idx][ao_output > 0] = np.minimum(min_cons_vals[cooling_idx], diff)[ao_output > 0]
        min_cons_vals[heating_idx][ao_output > 0] = np.minimum(min_cons_vals[heating_idx], diff)[ao_output > 0]

        min_cons_vals[get_index('li')][ao_output > 0] = np.minimum(min_cons_vals[get_index('li')], diff)[ao_output > 0]

        min_cons_vals = np.fmax(0, min_cons_vals)

    # maintain cooling ao/heating ao  ts level output same as disagg output,
    # incase of overshoot of total disagg values,
    # consumption is decreased on other appliances rather than hvac ao/high confidence PP users


    disagg_confidence = 0

    if (item_input_object.get('disagg_special_outputs') is not None) and \
            (item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') is not None):
        disagg_confidence = item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') / 100

    high_confidence_pp = disagg_confidence >= 0.75

    # maintaining a minimum cooling ao cons

    if ao_cool is not None:
        reduced_pp = input_data - ao_cool
        diff = input_data - ao_cool

        if disagg_confidence >= 0.75:
            diff = input_data - ao_cool - mid_cons_vals[pp_idx]
            reduced_pp = input_data - ao_cool - mid_cons_vals[pp_idx]

        min_cons_vals[cook_idx][ao_cool > 0] = np.minimum(min_cons_vals[cook_idx], diff)[ao_cool > 0]
        min_cons_vals[ld_idx][ao_cool > 0] = np.minimum(min_cons_vals[ld_idx], diff)[ao_cool > 0]
        min_cons_vals[ent_idx][ao_cool > 0] = np.minimum(min_cons_vals[ent_idx], diff)[ao_cool > 0]

        min_cons_vals[get_index('li')][ao_cool > 0] = np.minimum(min_cons_vals[get_index('li')], reduced_pp)[
            ao_cool > 0]

        if disagg_confidence >= 0.75:
            min_cons_vals[cooling_idx][ao_cool > 0] = \
            np.maximum(min_cons_vals[cooling_idx], np.minimum(ao_cool, input_data - mid_cons_vals[pp_idx] - ao_output))[
                ao_cool > 0]
            mid_cons_vals[cooling_idx][ao_cool > 0] = \
            np.maximum(mid_cons_vals[cooling_idx], np.minimum(ao_cool, input_data - mid_cons_vals[pp_idx]) - ao_output)[
                ao_cool > 0]
            max_cons_vals[cooling_idx][ao_cool > 0] = \
            np.maximum(max_cons_vals[cooling_idx], np.minimum(ao_cool, input_data - mid_cons_vals[pp_idx] - ao_output))[
                ao_cool > 0]
        else:
            min_cons_vals[cooling_idx][ao_cool > 0] = np.maximum(min_cons_vals[cooling_idx], ao_cool)[ao_cool > 0]
            mid_cons_vals[cooling_idx][ao_cool > 0] = np.maximum(mid_cons_vals[cooling_idx], ao_cool)[ao_cool > 0]
            max_cons_vals[cooling_idx][ao_cool > 0] = np.maximum(max_cons_vals[cooling_idx], ao_cool)[ao_cool > 0]

    # maintaining a minimum heating ao cons

    if ao_heat is not None:
        reduced_pp = input_data - ao_heat
        diff = input_data - ao_heat

        if high_confidence_pp:
            diff = input_data - ao_heat - mid_cons_vals[pp_idx]
            reduced_pp = input_data - ao_heat - mid_cons_vals[pp_idx]

        min_cons_vals[cook_idx][ao_heat > 0] = np.minimum(min_cons_vals[cook_idx], diff)[ao_heat > 0]
        min_cons_vals[ld_idx][ao_heat > 0] = np.minimum(min_cons_vals[ld_idx], diff)[ao_heat > 0]
        min_cons_vals[ent_idx][ao_heat > 0] = np.minimum(min_cons_vals[ent_idx], diff)[ao_heat > 0]

        cons_with_pp_and_ao = np.minimum(ao_heat, input_data - mid_cons_vals[pp_idx] - ao_output)

        # removing pp along with ao, if pp detection conf is high

        min_cons_vals[get_index('li')][ao_heat > 0] = np.minimum(min_cons_vals[get_index('li')], reduced_pp)[ao_heat > 0]

        if high_confidence_pp:
            min_cons_vals[heating_idx][ao_heat > 0] = \
            np.maximum(min_cons_vals[heating_idx], cons_with_pp_and_ao)[ao_heat > 0]
            mid_cons_vals[heating_idx][ao_heat > 0] = \
            np.maximum(mid_cons_vals[heating_idx], cons_with_pp_and_ao)[ao_heat > 0]
            max_cons_vals[heating_idx][ao_heat > 0] = \
            np.maximum(max_cons_vals[heating_idx], cons_with_pp_and_ao)[ao_heat > 0]

        else:
            min_cons_vals[heating_idx][ao_heat > 0] = np.maximum(min_cons_vals[heating_idx], ao_heat)[ao_heat > 0]
            mid_cons_vals[heating_idx][ao_heat > 0] = np.maximum(mid_cons_vals[heating_idx], ao_heat)[ao_heat > 0]
            max_cons_vals[heating_idx][ao_heat > 0] = np.maximum(max_cons_vals[heating_idx], ao_heat)[ao_heat > 0]

    return min_cons_vals, mid_cons_vals, max_cons_vals


def block_low_cons_hvac(item_input_object, item_output_object, output_data, min_cons_vals, mid_cons_vals, max_cons_vals, swh_pilots, logger):

    """
    Modify appliance mid/min/max ranges in cases where low consumption hvac months are present

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        output_data               (np.ndarray)    : disagg ts level output output
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        swh_pilots                (np.ndarray)    : list of swh pilots
        logger                    (logger)        : logger object

    Returns:
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
    """

    cooling_idx = get_index('cooling')
    heating_idx = get_index('heating')

    run_hybrid_v2 = item_input_object.get('item_input_params').get('run_hybrid_v2_flag')

    if run_hybrid_v2:

        # blocking low cons hvac months

        min_cons_vals, mid_cons_vals, max_cons_vals = \
            block_low_cons_cooling(item_input_object, item_output_object, min_cons_vals, mid_cons_vals, max_cons_vals,
                                   cooling_idx, output_data, swh_pilots, logger)

        min_cons_vals, mid_cons_vals, max_cons_vals = \
            block_low_cons_heating(item_input_object, item_output_object, min_cons_vals, mid_cons_vals, max_cons_vals,
                                   heating_idx, output_data, swh_pilots, logger)

    min_cons_required_in_opposite_season = 20000
    min_days_required_in_opposite_season = 15

    season = item_output_object.get("season")

    cooling_days = np.sum(output_data[cooling_idx], axis=1) > 0

    # blocking low duration hvac band detected in opposite season

    seq = find_seq(cooling_days, np.zeros_like(cooling_days), np.zeros_like(cooling_days), overnight=0)

    for i in range(len(seq)):
        if seq[i, 0] > 0 and seq[i, 3] < min_days_required_in_opposite_season and \
                np.sum(output_data[cooling_idx][seq[i, 1]:seq[i, 2]+1]) < min_cons_required_in_opposite_season and \
                np.all(season[seq[i, 1]:seq[i, 2]+1] < 0) and np.sum(output_data[cooling_idx]):
            mid_cons_vals[cooling_idx, seq[i, 1]:seq[i, 2] + 1] = 0
            min_cons_vals[cooling_idx, seq[i, 1]:seq[i, 2] + 1] = 0
            max_cons_vals[cooling_idx, seq[i, 1]:seq[i, 2] + 1] = 0

    # blocking low duration hvac band detected in opposite season

    cooling_days = np.sum(output_data[heating_idx], axis=1) > 0

    seq = find_seq(cooling_days, np.zeros_like(cooling_days), np.zeros_like(cooling_days), overnight=0)

    for i in range(len(seq)):
        if seq[i, 0] > 0 and seq[i, 3] < min_days_required_in_opposite_season and \
                np.sum(output_data[heating_idx][seq[i, 1]:seq[i, 2]+1]) < min_cons_required_in_opposite_season and \
                np.all(season[seq[i, 1]:seq[i, 2]+1] > 0) and np.sum(output_data[heating_idx]):
            mid_cons_vals[heating_idx, seq[i, 1]:seq[i, 2] + 1] = 0
            min_cons_vals[heating_idx, seq[i, 1]:seq[i, 2] + 1] = 0
            max_cons_vals[heating_idx, seq[i, 1]:seq[i, 2] + 1] = 0

    return min_cons_vals, mid_cons_vals, max_cons_vals


def block_low_cons_cooling(item_input_object, item_output_object, min_cons_vals, mid_cons_vals, max_cons_vals, cooling_idx, output_data, swh_pilots, logger):

    """
    blocking low cons cooling months

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        cooling_idx               (int)           : cooling index
        output_data               (np.ndarray)    : disagg ts level output output
        swh_pilots                (np.ndarray)    : list of swh pilots
        logger                    (logger)        : logger object

    Returns:
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
    """

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    disagg_cons = output_data[cooling_idx]

    block_cooling = np.zeros_like(unique_bc)

    # blocking added cooling output with less than 10kwh

    app_idx = cooling_idx
    c = 0

    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    pilot = item_input_object.get('config').get('pilot_id')

    thres = 5000 * (pilot in swh_pilots) + 10000 * (pilot not in swh_pilots)

    for i in range(len(unique_bc)):
        vacation_count_factor = np.sum(vacation_days[bc_list == unique_bc[i]]) / np.sum(bc_list == unique_bc[i])
        vacation_count_factor = 1 - vacation_count_factor

        block_cooling[i] = (np.sum(disagg_cons[bc_list == unique_bc[i]]) == 0) and\
                           not (np.sum(mid_cons_vals[app_idx, bc_list == unique_bc[i]]) > ((thres/30)*np.sum(bc_list == unique_bc[i]) )*vacation_count_factor )

        c = c + bool(np.sum(disagg_cons[bc_list == unique_bc[i]]))

    for i in range(len(unique_bc)):
        mid_cons_vals[app_idx][bc_list == unique_bc[i]] = mid_cons_vals[app_idx][bc_list == unique_bc[i]] * (1-block_cooling[i])
        max_cons_vals[app_idx][bc_list == unique_bc[i]] = max_cons_vals[app_idx][bc_list == unique_bc[i]] * (1-block_cooling[i])
        min_cons_vals[app_idx][bc_list == unique_bc[i]] = min_cons_vals[app_idx][bc_list == unique_bc[i]] * (1-block_cooling[i])
        logger.info('blocked cooling for billing cycle | %s', unique_bc[i])

    # blocking low consumption cooling output, based on the threshold given in pilot config

    detected_cool = item_output_object.get("hvac_dict").get("cooling")
    disagg_cons = output_data[cooling_idx]

    block_cooling = np.zeros_like(unique_bc)

    app_idx = cooling_idx
    c = 0

    cooling_limit = item_input_object.get('pilot_level_config').get('cooling_config').get('bounds').get('block_if_less_than') * 1000

    for i in range(len(unique_bc)):

        # before blocking consumption is scaled based on vacation days present

        vacation_count_factor = np.sum(vacation_days[bc_list == unique_bc[i]]) / np.sum(bc_list == unique_bc[i])
        vacation_count_factor = 1 - vacation_count_factor

        if (np.sum(disagg_cons[bc_list == unique_bc[i]]) < ((cooling_limit/30)*np.sum(bc_list == unique_bc[i]))*vacation_count_factor) and \
                not (np.sum(detected_cool[bc_list == unique_bc[i]]) > ((30000/30)*np.sum(bc_list == unique_bc[i]))):

            block_cooling[i] = 1

        if np.sum(disagg_cons[bc_list == unique_bc[i]]):
            c = c + 1

    seq = find_seq(block_cooling,np.zeros_like(block_cooling), np.zeros_like(block_cooling), overnight=0)

    for i in range(len(seq)):
        if (seq[i, 0] and seq[i, 3] < 2) and c > 1:
            block_cooling[int(seq[i, 1])] = 0

    for i in range(len(unique_bc)):

        mid_cons_vals[app_idx][bc_list == unique_bc[i]] = mid_cons_vals[app_idx][bc_list == unique_bc[i]] * (1-block_cooling[i])
        max_cons_vals[app_idx][bc_list == unique_bc[i]] = max_cons_vals[app_idx][bc_list == unique_bc[i]] * (1-block_cooling[i])
        min_cons_vals[app_idx][bc_list == unique_bc[i]] = min_cons_vals[app_idx][bc_list == unique_bc[i]] * (1-block_cooling[i])

        logger.info('blocked cooling for billing cycle | %s', unique_bc[i])

    return min_cons_vals, mid_cons_vals, max_cons_vals


def block_low_cons_heating(item_input_object, item_output_object, min_cons_vals, mid_cons_vals, max_cons_vals, heating_idx, output_data, swh_pilots, logger):

    """
    blocking low cons cooling months

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        cooling_idx               (int)           : cooling index
        output_data               (np.ndarray)    : disagg ts level output output
        swh_pilots                (np.ndarray)    : list of swh pilots
        logger                    (logger)        : logger object

    Returns:
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
    """

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    thres = 10000

    # blocking added heating output with less than 10kwh

    disagg_cons = output_data[heating_idx]
    app_idx = heating_idx
    block_heating = np.zeros_like(unique_bc)
    c = 0

    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    for i in range(len(unique_bc)):

        vacation_count_factor = np.sum(vacation_days[bc_list == unique_bc[i]]) / np.sum(bc_list == unique_bc[i])
        vacation_count_factor = 1 - vacation_count_factor

        block_heating[i] = (np.sum(disagg_cons[bc_list == unique_bc[i]]) == 0) and \
                           not (np.sum(mid_cons_vals[app_idx, bc_list == unique_bc[i]]) > ((thres / 30) * np.sum(bc_list == unique_bc[i]))*vacation_count_factor)

        c = c + bool(np.sum(disagg_cons[bc_list == unique_bc[i]]))

    seq = find_seq(block_heating, np.zeros_like(block_heating), np.zeros_like(block_heating), overnight=0)

    for i in range(len(seq)):
        if (seq[i, 0] and seq[i, 3] < 2) and c > 1:
            block_heating[int(seq[i, 1])] = 0

    for i in range(len(unique_bc)):
        mid_cons_vals[app_idx][bc_list == unique_bc[i]] = mid_cons_vals[app_idx][bc_list == unique_bc[i]] * (1 - block_heating[i])
        max_cons_vals[app_idx][bc_list == unique_bc[i]] = max_cons_vals[app_idx][bc_list == unique_bc[i]] * (1 - block_heating[i])
        min_cons_vals[app_idx][bc_list == unique_bc[i]] = min_cons_vals[app_idx][bc_list == unique_bc[i]] * (1 - block_heating[i])
        logger.info('blocked heating for billing cycle | %s', unique_bc[i])

    # blocking low consumption cooling output, based on the threshold given in pilot config

    disagg_cons = copy.deepcopy(output_data[heating_idx])

    app_idx = heating_idx
    block_heating = np.zeros_like(unique_bc)
    detected_heat = item_output_object.get("hvac_dict").get("heating")

    heating_limit = item_input_object.get('pilot_level_config').get('heating_config').get('bounds').get('block_if_less_than') * 1000

    for i in range(len(unique_bc)):

        # before blocking consumption is scaled based on vacation days present

        vacation_count_factor = np.sum(vacation_days[bc_list == unique_bc[i]]) / np.sum(bc_list == unique_bc[i])
        vacation_count_factor = 1 - vacation_count_factor

        if (np.sum(disagg_cons[bc_list == unique_bc[i]]) < ((heating_limit / 30) * np.sum(bc_list == unique_bc[i]) )*vacation_count_factor ) and \
                not (np.sum(detected_heat[bc_list == unique_bc[i]]) > ((30000 / 30) * np.sum(bc_list == unique_bc[i]))):
            block_heating[i] = 1

    seq = find_seq(block_heating, np.zeros_like(block_heating), np.zeros_like(block_heating), overnight=0)

    for i in range(len(seq)):
        if (seq[i, 0] and seq[i, 3] < 2):
            block_heating[int(seq[i, 1])] = 0

    for i in range(len(unique_bc)):

        mid_cons_vals[app_idx][bc_list == unique_bc[i]] = mid_cons_vals[app_idx][bc_list == unique_bc[i]] * (1 - block_heating[i])
        max_cons_vals[app_idx][bc_list == unique_bc[i]] = max_cons_vals[app_idx][bc_list == unique_bc[i]] * (1 - block_heating[i])
        min_cons_vals[app_idx][bc_list == unique_bc[i]] = min_cons_vals[app_idx][bc_list == unique_bc[i]] * (1 - block_heating[i])

        logger.info('blocked heating for billing cycle | %s', unique_bc[i])

    return min_cons_vals, mid_cons_vals, max_cons_vals

