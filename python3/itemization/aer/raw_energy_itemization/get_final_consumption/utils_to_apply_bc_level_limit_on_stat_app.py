
"""
Author - Nisha Agarwal
Date - 4th April 2021
This file has functions that is used to maintain a billing cycle level min/max consumption for appliances after adjustment
"""

# Import python packages

import copy
import numpy as np
import pandas as pd
from numpy.random import RandomState

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.init_itemization_config import random_gen_config

from python3.itemization.aer.functions.get_config import get_hybrid_config

from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.hsm_utils import get_backup_app_hsm

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.init_final_item_config import init_final_item_conf

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import get_app_idx
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import add_randomness
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import prepare_stat_tou
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import get_app_cons_in_target_bc
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import get_stat_app_min_cons_hard_limit
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import prepare_ts_level_additional_cons_points
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import add_prepare_box_type_additional_cons_points
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import add_step3_cons_into_stat_to_maintain_stability
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import prepare_additional_cons_points_to_satisfy_min_cons


from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.pilot_constants import PilotConstants
from python3.itemization.init_itemization_config import random_gen_config

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.get_hybrid_v2_generic_config import get_hybrid_v2_generic_config

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.get_config import get_hybrid_config

from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.init_final_item_config import init_final_item_conf

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.hsm_utils import get_backup_app_hsm


def apply_max_limit_on_ref_li(item_input_object, appliance_list, final_tou_consumption, processed_input_data):

    """
    Limit max bc level level consumption for all appliances based on hsm and pilot config

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        processed_input_data        (np.ndarray)    : user input data

    Returns:
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
    """

    config = get_hybrid_v2_generic_config()

    max_limit = item_input_object.get('pilot_level_config').get('li_config').get('bounds').get('max_cons')

    # applying max limit on billing cycle consumption for lighting

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc = np.unique(bc_list)

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    app = 'li'

    for bc in unique_bc:

        target_days = bc_list == bc

        if np.sum(target_days) < config.get('min_days_required_for_max_limit'):
            continue

        # scale monthly consumption based on vacation days count

        vac_factor = np.sum(vacation[target_days]) / np.sum(target_days)
        vac_factor = 1 - vac_factor

        app_idx = get_app_idx(appliance_list, app)

        factor = (final_tou_consumption[app_idx][target_days].sum() / (max_limit * vac_factor * Cgbdisagg.WH_IN_1_KWH)) * \
                 Cgbdisagg.DAYS_IN_MONTH / np.sum(target_days)

        if factor > 1:
            final_tou_consumption[app_idx][target_days] = final_tou_consumption[app_idx][target_days] / factor

    max_limit = item_input_object.get('pilot_level_config').get('ref_config').get('bounds').get('max_cons')

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    unique_bc = unique_bc[counts > 5]

    disconnection_days = np.sum(processed_input_data > 0, axis=1) == 0

    # applying max limit on billing cycle consumption for ref

    app = 'ref'

    for bc in unique_bc:

        target_days = bc_list == bc

        season_val = 1

        if np.sum(target_days) < config.get('min_days_required_for_max_limit'):
            continue

        vac_factor = np.sum(disconnection_days[target_days]) / np.sum(target_days)
        vac_factor = 1 - vac_factor

        vac_factor = vac_factor * season_val

        app_idx = get_app_idx(appliance_list, app)

        # this scaling factor is used to scale down app cons to max consumption allowed at billing cycle level

        factor = (final_tou_consumption[app_idx][target_days].sum() / (max_limit * Cgbdisagg.WH_IN_1_KWH * vac_factor)) * \
                 Cgbdisagg.DAYS_IN_MONTH / np.sum(target_days)

        if factor > 1:
            final_tou_consumption[app_idx][target_days] = final_tou_consumption[app_idx][target_days] / factor

    return final_tou_consumption


def apply_max_limit_on_step3_app(item_input_object, appliance_list, final_tou_consumption, logger):

    """
    Limit max bc level level consumption for all appliances based on hsm and pilot config

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        logger                      (logger)        : logger object

    Returns:
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
    """

    config = get_hybrid_v2_generic_config()

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc = np.unique(bc_list)

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    # applying max limit on billing cycle consumption for pp/ev/wh if added through step 3 addition (statistical addition)

    use_hsm, hsm_output = get_backup_app_hsm(item_input_object, item_input_object["item_input_params"]["backup_app"],
                                             logger)

    sig_based_app = item_input_object["item_input_params"]["backup_app"]

    for bc in unique_bc:

        target_days = bc_list == bc

        if np.sum(target_days) < config.get('min_days_required_for_max_limit'):
            continue

        for i, app in enumerate(sig_based_app):

            vac_factor = 1

            # scale monthly consumption based on vacation days count

            if app != 'pp':
                vac_factor = np.sum(vacation[target_days]) / np.sum(target_days)
                vac_factor = 1 - vac_factor

            max_limit = item_input_object.get('pilot_level_config').get(app + '_config').get('bounds').get('max_cons')

            app_idx = get_app_idx(appliance_list, app)

            hsm_idx = np.where(np.array(item_input_object["item_input_params"]["backup_app"]) == app)[0][0]

            if use_hsm[hsm_idx]:
                max_limit = min(max_limit, hsm_output[hsm_idx] * 1.1)

            # this scaling factor is used to scale down app cons to max consumption allowed at billing cycle level

            factor = (final_tou_consumption[app_idx][target_days].sum() / (max_limit * vac_factor * Cgbdisagg.WH_IN_1_KWH)) * \
                     Cgbdisagg.DAYS_IN_MONTH / np.sum(target_days)

            if factor > 1:
                final_tou_consumption[app_idx][target_days] = final_tou_consumption[app_idx][target_days] / factor

    return final_tou_consumption


def apply_bc_level_max_limit_based_on_config_and_hsm(item_input_object, item_output_object, final_tou_consumption,
                                                     app_month_cons, stat_hsm_output, logger):

    """
    Limit max bc level level consumption for all appliances based on hsm and pilot config

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        app_month_cons              (list)          : monthly stat app consumption
        stat_hsm_output             (list)          : monthly stat app consumption based on HSM
        logger                      (logger)        : logger object

    Returns:
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        ld_monthly_cons             (float)         : monthly laundry consumption
        ent_monthly_cons            (float)         : monthly ent consumption
        cook_monthly_cons           (float)         : monthly cooking consumption
    """

    ld_monthly_cons = app_month_cons[0]
    ent_monthly_cons = app_month_cons[1]
    cook_monthly_cons = app_month_cons[2]

    hsm_cook = stat_hsm_output[0]
    hsm_ent = stat_hsm_output[1]
    hsm_ld = stat_hsm_output[2]

    processed_input_data = item_output_object.get('original_input_data')
    appliance_list = np.array(item_input_object.get("item_input_params").get('app_list'))

    # default max consumption for cook/ld/ent from pilot config

    config = get_hybrid_v2_generic_config()

    cook_limit = config.get('cook_limit')
    ld_limit = config.get('ld_limit')
    ent_limit = config.get('ent_limit')

    stat_app_list = ['cook', 'ent', 'ld']
    cons_limit_arr = [cook_limit, ent_limit, ld_limit]

    # fetching max consumption for cook/ld/ent from pilot config

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    ld_idx = hybrid_config.get("app_seq").index('ld')
    have_max_cons = hybrid_config.get("have_max_cons")[ld_idx]
    max_cons = hybrid_config.get("max_cons")[ld_idx]

    if have_max_cons:
        cons_limit_arr[2] = min(cons_limit_arr[2], max_cons)

    ld_idx = hybrid_config.get("app_seq").index('cook')
    have_max_cons = hybrid_config.get("have_max_cons")[ld_idx]
    max_cons = hybrid_config.get("max_cons")[ld_idx]

    if have_max_cons:
        cons_limit_arr[0] = min(cons_limit_arr[0], max_cons)

    ld_idx = hybrid_config.get("app_seq").index('ent')
    have_max_cons = hybrid_config.get("have_max_cons")[ld_idx]
    max_cons = hybrid_config.get("max_cons")[ld_idx]

    if have_max_cons:
        cons_limit_arr[1] = min(cons_limit_arr[1], max_cons)

    use_stat_app_hsm_flag = (item_input_object.get('config').get('disagg_mode') == 'mtd') or \
                            ((item_input_object.get('config').get('disagg_mode') == 'incremental') and len(final_tou_consumption[0]) > 200)

    # updating max consumption for cook/ld/ent based on previous runs output, to maintain consistency

    if use_stat_app_hsm_flag:
        cons_limit_arr[0] = min(cons_limit_arr[0], hsm_cook)
        cons_limit_arr[1] = min(cons_limit_arr[1], hsm_ent)
        cons_limit_arr[2] = min(cons_limit_arr[2], hsm_ld)

    # applying max limit on billing cycle consumption for cook/ent/ld appliances

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc = np.unique(bc_list)

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    for i, app in enumerate(stat_app_list):

        for bc in unique_bc:

            target_days = bc_list == bc

            if np.sum(target_days) < config.get('min_days_required_for_max_limit'):
                continue

            # scale monthly consumption based on vacation days count

            season_val = 1 * (1 - (vacation[bc_list == bc].sum()/ np.sum(bc_list == bc)))

            app_idx = get_app_idx(appliance_list, app)

            # this scaling factor is used to scale down app cons to max consumption allowed at billing cycle level

            factor = (final_tou_consumption[app_idx][target_days].sum() / (season_val * cons_limit_arr[i] * Cgbdisagg.WH_IN_1_KWH)) *\
                     Cgbdisagg.DAYS_IN_MONTH / np.sum(target_days)

            if factor > 1:
                final_tou_consumption[app_idx][target_days] = final_tou_consumption[app_idx][target_days] / factor

    final_tou_consumption = apply_max_limit_on_ref_li(item_input_object, appliance_list, final_tou_consumption, processed_input_data)

    final_tou_consumption = apply_max_limit_on_step3_app(item_input_object, appliance_list, final_tou_consumption, logger)

    return final_tou_consumption, ld_monthly_cons, ent_monthly_cons, cook_monthly_cons


def apply_bc_level_max_limit_based_on_config(item_input_object, item_output_object, final_tou_consumption, length,
                                             app_month_cons, stat_hsm_output, total_monthly_cons):
    """
    Limit max bc level level consumption for statistical based appliances

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        length                      (int)           : total non vacation days of the user
        app_month_cons              (list)          : monthly stat app consumption
        stat_hsm_output             (list)          : monthly stat app consumption based on HSM
        total_monthly_cons          (float)         : monthly total consumption

    Returns:
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        ld_monthly_cons             (float)         : monthly laundry consumption
        ent_monthly_cons            (float)         : monthly ent consumption
        cook_monthly_cons           (float)         : monthly cooking consumption
    """

    ld_monthly_cons = app_month_cons[0]
    ent_monthly_cons = app_month_cons[1]
    cook_monthly_cons = app_month_cons[2]

    hsm_cook = stat_hsm_output[0]
    hsm_ent = stat_hsm_output[1]
    hsm_ld = stat_hsm_output[2]

    processed_input_data = item_output_object.get('original_input_data')
    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    appliance_list = np.array(item_input_object.get("item_input_params").get('app_list'))
    ld_cons = final_tou_consumption[np.where(appliance_list == 'ld')[0][0]][np.logical_not(vacation)]
    ent_cons = final_tou_consumption[np.where(appliance_list == 'ent')[0][0]][np.logical_not(vacation)]
    cook_cons = final_tou_consumption[np.where(appliance_list == 'cook')[0][0]][np.logical_not(vacation)]

    scaling_factor = Cgbdisagg.DAYS_IN_MONTH / 1000

    cook_cons = ((np.sum(cook_cons) / length) * (scaling_factor))
    ent_cons = ((np.sum(ent_cons) / length) * (scaling_factor))
    ld_cons = ((np.sum(ld_cons) / length) * (scaling_factor))

    # fetching max consumption for cook/ld/ent based on total consumption level

    config = init_final_item_conf(total_monthly_cons).get('min_max_limit_conf')

    ent_limit = config.get('ent_max_limit')
    cook_limit = config.get('cook_max_limit')
    ld_limit = config.get('ld_max_limit')

    # Modify limit based on app profile information

    if item_input_object.get("appliance_profile").get("cooking_type")[0] == 0:
        cook_limit = min(30, cook_limit * 0.7)

    if item_input_object.get("appliance_profile").get("cooking_type")[1] == 0:
        cook_limit = min(30, cook_limit * 0.7)

    if np.sum(item_input_object.get("appliance_profile").get("laundry_type")) != 3:
        ld_limit = ld_limit * 0.9

    config = get_hybrid_v2_generic_config()

    cook_limit = min(config.get('cook_limit'), cook_limit)
    ld_limit = min(config.get('ld_limit'), ld_limit)
    ent_limit = min(config.get('ent_limit'), ent_limit)

    if len(final_tou_consumption[0]) <= 70:
        cook_limit = min(config.get('cook_limit') * 0.5, cook_limit)
        ld_limit = min(config.get('ld_limit') * 0.5, ld_limit)
        ent_limit = min(config.get('ent_limit') * 0.5, ent_limit)

    # fetching max consumption for cook/ld/ent based on total consumption level

    stat_app_array = [ld_monthly_cons, ent_monthly_cons, cook_monthly_cons]
    dev_qa_idx_arr = [2, 1, 0]
    stat_app_list = ['cook', 'ent', 'ld']
    bc_level_cons = [cook_cons, ent_cons, ld_cons]
    cons_limit_arr = [cook_limit, ent_limit, ld_limit]

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    ld_idx = hybrid_config.get("app_seq").index('ld')
    have_max_cons = hybrid_config.get("have_max_cons")[ld_idx]
    max_cons = hybrid_config.get("max_cons")[ld_idx]

    if have_max_cons:
        cons_limit_arr[2] = min(cons_limit_arr[2], max_cons)

    ld_idx = hybrid_config.get("app_seq").index('cook')
    have_max_cons = hybrid_config.get("have_max_cons")[ld_idx]
    max_cons = hybrid_config.get("max_cons")[ld_idx]

    if have_max_cons:
        cons_limit_arr[0] = min(cons_limit_arr[0], max_cons)

    ld_idx = hybrid_config.get("app_seq").index('ent')
    have_max_cons = hybrid_config.get("have_max_cons")[ld_idx]
    max_cons = hybrid_config.get("max_cons")[ld_idx]

    if have_max_cons:
        cons_limit_arr[1] = min(cons_limit_arr[1], max_cons)

    use_stat_app_hsm_flag = (item_input_object.get('config').get('disagg_mode') == 'mtd') or \
                            ((item_input_object.get('config').get('disagg_mode') == 'incremental') and len(final_tou_consumption[0]) > 200)

    if use_stat_app_hsm_flag:
        cons_limit_arr[0] = min(cons_limit_arr[0], hsm_cook)
        cons_limit_arr[1] = min(cons_limit_arr[1], hsm_ent)
        cons_limit_arr[2] = min(cons_limit_arr[2], hsm_ld)

    # applying max limit on billing cycle consumption for cook/ent/ld appliances

    for i, app in enumerate(stat_app_list):

        factor = bc_level_cons[i] / cons_limit_arr[i]

        if factor > 1:
            app_idx = get_app_idx(appliance_list, app)
            final_tou_consumption[app_idx] = final_tou_consumption[app_idx] / factor
            final_tou_consumption[app_idx] = np.minimum(final_tou_consumption[app_idx], processed_input_data)
            cons = final_tou_consumption[app_idx][np.logical_not(vacation)]
            stat_app_array[dev_qa_idx_arr[i]] = ((np.sum(cons) / length) * scaling_factor)

    return final_tou_consumption, stat_app_array[0], stat_app_array[1], stat_app_array[2]


def bc_level_min_limit_using_step3_app(final_tou_consumption, item_input_object, item_output_object, input_dict_for_min_cons):

    """
    maintain bc level level consistency of ld, cook, and ent appliances by picking slight consumption from other appliances

    Parameters:
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        input_dict_for_min_cons     (dict)          : additional inputs required for maintain bc level min cons

    Returns:
        final_tou_consumption       (np.ndarray)    : updated final ts level estimation of appliances
    """

    inactive_hours = input_dict_for_min_cons.get('inactive_hours')
    app = input_dict_for_min_cons.get('app')
    max_cons = input_dict_for_min_cons.get('max_cons')
    target_days = input_dict_for_min_cons.get('target_days')
    stat_tou = input_dict_for_min_cons.get('stat_tou')
    app_cons = input_dict_for_min_cons.get('app_cons')
    hard_limit = input_dict_for_min_cons.get('hard_limit')

    backup_app_list = item_input_object.get("item_input_params").get("backup_app")
    processed_input_data = item_output_object.get('original_input_data')
    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    appliance_list = np.array(item_input_object.get("item_input_params").get('app_list'))

    app_idx = get_app_idx(appliance_list, app)

    for backup_app in backup_app_list:

        active_hours = np.ones_like(stat_tou)
        active_hours[inactive_hours] = 0

        # picking required consumption from step3 consumption(statistical output of PP/EV/WH)
        # A limited amount of monthly consumption is added from step3 to stat app
        # to maintain consistency at billing cycle level output

        if (app_cons < (hard_limit - 1)) and (app_cons >= 0):
            backup_id = np.where(appliance_list == backup_app)[0][0]

            final_tou_consumption, continue_flag = \
                add_step3_cons_into_stat_to_maintain_stability([final_tou_consumption, processed_input_data],
                                                               active_hours, target_days, backup_id, max_cons,
                                                               hard_limit, app_idx, vacation, app_cons, 0.15)

    return final_tou_consumption


def apply_bc_level_min_limit_for_step3_app(total_cons, item_input_object, item_output_object, final_tou_consumption, logger):

    """
    add billing cycle min cons limit for statistical PP/EV/WH

    Parameters:
        total_cons                  (float)         : total consumption
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        logger                      (logger)        : logger object

    Returns:
        final_tou_consumption       (np.ndarray)    : updated final ts level estimation of appliances
    """

    processed_input_data = item_output_object.get('original_input_data')
    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')
    scaling_factor = Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH

    day_start_hour = 6

    total = (processed_input_data[np.logical_not(vacation)][day_start_hour * samples_per_hour:]) * \
            Cgbdisagg.HRS_IN_DAY / (Cgbdisagg.HRS_IN_DAY - day_start_hour)
    total = ((np.sum(total) / len(processed_input_data)) * (scaling_factor))

    limit_cons = (total - 5000) / 200 + 50
    limit_cons = min(90, limit_cons)
    limit_cons = max(limit_cons, 2)

    limit_cons = min(limit_cons, [8, 8, 15, 20, 25, 30, 40, 50, 70, 80][np.digitize(total_cons, [300, 400, 700, 1000, 1500, 2000, 3000, 4000, 6000])])

    ent_limit = limit_cons
    cook_limit = limit_cons * 0.8
    ld_limit = limit_cons * 1.3

    cons_limit_arr = [ld_limit, cook_limit, ent_limit]

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    ld_idx = hybrid_config.get("app_seq").index('ld')
    have_min_cons = hybrid_config.get("have_hard_min_lim")[ld_idx]
    min_cons = hybrid_config.get("hard_min_lim")[ld_idx]

    if have_min_cons:
        cons_limit_arr[0] = max(cons_limit_arr[0], min_cons)

    if have_min_cons:
        cons_limit_arr[2] = max(cons_limit_arr[2], min_cons)

    backup_app_list = item_input_object["item_input_params"]["backup_app"]
    appliance_list = np.array(item_input_object.get("item_input_params").get('app_list'))

    stat_app_list = backup_app_list

    use_hsm, hsm_output = get_backup_app_hsm(item_input_object, item_input_object["item_input_params"]["backup_app"], logger)

    sleep_hours = copy.deepcopy(item_output_object.get("profile_attributes").get("sleep_hours"))

    date_list = item_output_object.get("date_list")
    month_list = pd.DatetimeIndex(date_list).month.values

    app_season = np.zeros((len(appliance_list), 12))
    app_season[get_app_idx(appliance_list, 'ref')] = [-0.15, -0.15, -0.1, -0.1, -0.05, -0.05, 0, 0, -0.05, -0.05, -0.1, -0.15]
    app_season[get_app_idx(appliance_list, 'wh')] = [0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.2, -0.15, -0.1, -0.05, 0]
    app_season[get_app_idx(appliance_list, 'li')] = [0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.25, -0.2, -0.15, -0.1, -0.05, 0]
    app_season[get_app_idx(appliance_list, 'cooling')] = [-0.6, -0.6, -0.4, -0.3, -0.2, 0, 0, 0, -0.2, -0.3, -0.4, -0.6]
    app_season[get_app_idx(appliance_list, 'heating')] = [0, 0, -0.2, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.2, 0, 0]

    if np.all(sleep_hours == 0):
        sleep_hours = np.ones_like(sleep_hours)
        sleep_hours[np.arange(1, 5 * samples_per_hour+1)] = 0

    for j, app in enumerate(stat_app_list):

        if np.sum(final_tou_consumption[get_app_idx(appliance_list, app)]) == 0:
            continue

        unique_bc, days_count = np.unique(bc_list, return_counts=1)

        monthly_cons = np.zeros(len(unique_bc))

        # calculate appliance consumption at bill cycle level

        for idx, current_bc in enumerate(unique_bc):

            if np.sum(np.sum(bc_list == current_bc)) <= 8:
                continue

            monthly_cons[idx] = np.sum(
                final_tou_consumption[get_app_idx(appliance_list, app)][bc_list == current_bc])

            monthly_cons[idx] = monthly_cons[idx] * (Cgbdisagg.DAYS_IN_MONTH / np.sum(bc_list == current_bc))

            if (1 - (np.sum(vacation[bc_list == current_bc]) / np.sum(bc_list == current_bc))) != 0:
                monthly_cons[idx] = monthly_cons[idx] / (1 - (np.sum(vacation[bc_list == current_bc]) / np.sum(bc_list == current_bc)))
            else:
                monthly_cons[idx] = 0

        app_idx = get_app_idx(appliance_list, app)

        unique_bc, days_count = np.unique(bc_list, return_counts=1)
        unique_bc = unique_bc[days_count >= 5]
        unique_bc = unique_bc[unique_bc > 0]

        for current_bc in unique_bc:

            target_days = bc_list == current_bc

            input_dict_for_min_cons = {
                'target_days': target_days,
                'app': app,
                'target_idx': j,
                'hsm_output': hsm_output,
                'use_hsm': use_hsm
            }

            final_tou_consumption, disagg_cons = \
                apply_bc_level_min_limit_for_step3_app_for_each_bc(item_input_object, item_output_object, monthly_cons,
                                                                   final_tou_consumption, input_dict_for_min_cons)

            season_val = app_season[app_idx][(month_list[bc_list == current_bc] - 1).astype(int)].mean() + 1
            final_tou_consumption[get_app_idx(appliance_list, app)][bc_list == current_bc] = \
                final_tou_consumption[get_app_idx(appliance_list, app)][bc_list == current_bc] * season_val

        final_tou_consumption[app_idx][vacation] = 0

    return final_tou_consumption


def apply_bc_level_min_limit_for_step3_app_for_each_bc(item_input_object, item_output_object, monthly_cons,
                                                       final_tou_consumption, input_dict_for_min_cons):

    """
    maintain bc level level consistency of ld, cook, and ent appliances by picking slight consumption from other appliances

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        monthly_cons                (np.ndarray)    : final ts level estimation of appliances
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        input_dict_for_min_cons     (dict)          : additional inputs required for maintain bc level min cons

    Returns:
        final_tou_consumption       (np.ndarray)    : updated final ts level estimation of appliances
    """

    use_hsm = input_dict_for_min_cons.get('use_hsm')
    app = input_dict_for_min_cons.get('app')
    target_idx = input_dict_for_min_cons.get('target_idx')
    target_days = input_dict_for_min_cons.get('target_days')
    hsm_output = input_dict_for_min_cons.get('hsm_output')

    processed_input_data = item_output_object.get('original_input_data')
    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    appliance_list = np.array(item_input_object.get("item_input_params").get('app_list'))

    cool_idx = np.where(appliance_list == "cooling")[0][0]
    heat_idx = np.where(appliance_list == "heating")[0][0]

    cooling_limit = item_input_object.get('pilot_level_config').get('cooling_config').get('bounds').get('min_cons')
    heating_limit = item_input_object.get('pilot_level_config').get('heating_config').get('bounds').get('min_cons')

    disagg_cons = final_tou_consumption[cool_idx] + \
                  final_tou_consumption[heat_idx]

    if item_input_object.get("item_input_params").get("ao_cool") is not None:
        disagg_cons = disagg_cons - item_input_object.get("item_input_params").get("ao_cool")

    if item_input_object.get("item_input_params").get("ao_heat") is not None:
        disagg_cons = disagg_cons - item_input_object.get("item_input_params").get("ao_heat")

    disagg_cons = np.fmax(0, disagg_cons)

    disagg_thres = 0.03
    max_cons = 5000
    min_stat_cons = 500

    app_idx = get_app_idx(appliance_list, app)

    app_cons = get_app_cons_in_target_bc(final_tou_consumption, vacation, target_days, app_idx)

    hard_limit = np.median(monthly_cons[monthly_cons > 0]) / Cgbdisagg.WH_IN_1_KWH

    if use_hsm[target_idx]:
        hard_limit = max(hard_limit, hsm_output[target_idx] * 0.9)

    if np.sum(final_tou_consumption[cool_idx]) < cooling_limit * 1.1:
        disagg_cons[target_days] = disagg_cons[target_days] - final_tou_consumption[cool_idx][target_days]

    if np.sum(final_tou_consumption[heat_idx]) < heating_limit * 1.1:
        disagg_cons[target_days] = disagg_cons[target_days] - final_tou_consumption[heat_idx][target_days]

    disagg_cons = np.fmax(0, disagg_cons)

    if (app_cons < hard_limit) and (app_cons > 0):

        score = np.zeros_like(processed_input_data[target_days])

        # picking required monthly consumption from residual data

        other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)
        other_cons_arr = np.fmax(0, other_cons_arr)
        score[:] = np.fmin(max_cons, other_cons_arr[target_days]) / (
            np.max(np.fmin(max_cons, other_cons_arr[target_days][:])))
        score = np.nan_to_num(score)

        score = add_randomness(score)
        score[vacation[target_days]] = 0

        samples = int(processed_input_data.shape[1] / Cgbdisagg.HRS_IN_DAY)
        score[processed_input_data[target_days] < min_stat_cons / samples] = 0

        if np.sum(np.nan_to_num(score)) > 0:
            score[vacation[target_days]] = 0

            score = prepare_additional_cons_points_to_satisfy_min_cons(score, hard_limit, target_days, app_cons)

            other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

            final_tou_consumption[app_idx, target_days, :] = \
                final_tou_consumption[app_idx, target_days, :] + np.fmax(0, np.minimum(other_cons_arr[target_days, :] * 0.99, score).astype(int))

            final_tou_consumption[app_idx, target_days, :] = np.minimum(final_tou_consumption[app_idx, target_days, :], processed_input_data[target_days])

            final_tou_consumption[app_idx, target_days, :] = np.fmin(final_tou_consumption[app_idx, target_days, :], max_cons)

            app_cons = get_app_cons_in_target_bc(final_tou_consumption, vacation, target_days, app_idx)

            app_cons = np.fmax(0, app_cons)

    if (app_cons < (hard_limit - 1)) and (app_cons >= 0):

        # picking required monthly consumption from HVAC, with a limit on max consumption being taken from hvac

        score = np.zeros_like(processed_input_data[target_days])

        pick_cons = min(disagg_thres * (disagg_cons[target_days].sum() / Cgbdisagg.WH_IN_1_KWH),
                        ((hard_limit - app_cons) * (np.sum(target_days) / Cgbdisagg.DAYS_IN_MONTH)) * 0.9) * Cgbdisagg.WH_IN_1_KWH
        score[:] = np.fmin(max_cons, disagg_cons[target_days][:]) / np.max(
            np.fmin(max_cons, disagg_cons[target_days][:]))

        if not (np.max(disagg_cons[target_days][:]) == 0):
            score = prepare_ts_level_additional_cons_points(score, vacation, pick_cons, target_days)

            final_tou_consumption[app_idx, target_days, :] = \
                final_tou_consumption[app_idx, target_days, :] + \
                np.minimum(processed_input_data[target_days], np.minimum(disagg_cons[target_days], score))

            final_tou_consumption[app_idx, target_days, :] = np.minimum(final_tou_consumption[app_idx, target_days, :], processed_input_data[target_days])

            final_tou_consumption[cool_idx, target_days, :] = \
                np.fmax(0, final_tou_consumption[cool_idx, target_days, :] - np.fmax(0, np.minimum(disagg_cons[target_days], score)))

            final_tou_consumption[heat_idx, target_days, :] = \
                np.fmax(0, final_tou_consumption[heat_idx, target_days, :] - np.fmax(0, np.minimum(disagg_cons[target_days], score)))

            disagg_cons[target_days] = disagg_cons[target_days] - np.minimum(disagg_cons[target_days], score)

    return final_tou_consumption, disagg_cons


def apply_bc_level_min_limit_using_hvac_cons(final_tou_consumption, item_input_object, item_output_object, input_dict_for_min_cons):

    """
    maintain bc level level consistency of ld, cook, and ent appliances by picking slight consumption from other appliances

    Parameters:
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        input_dict_for_min_cons     (dict)          : additional inputs required for maintain bc level min cons

    Returns:
        final_tou_consumption       (np.ndarray)    : updated final ts level estimation of appliances
        app_cons                    (int)           : current bc level cons of the appliance
        disagg_cons                 (np.ndarray)    : amount of hvac app that can be partially alloted to stat app to maintain consistency
        hvac_cons_val               (np.ndarray)    : ts level hvac cons
    """

    inactive_hours = input_dict_for_min_cons.get('inactive_hours')
    app = input_dict_for_min_cons.get('app')
    max_cons = input_dict_for_min_cons.get('max_cons')
    target_days = input_dict_for_min_cons.get('target_days')
    stat_tou = input_dict_for_min_cons.get('stat_tou')
    app_cons = input_dict_for_min_cons.get('app_cons')
    hard_limit = input_dict_for_min_cons.get('hard_limit')
    disagg_cons = input_dict_for_min_cons.get('disagg_cons')
    hvac_cons_val = input_dict_for_min_cons.get('hvac_cons_val')

    processed_input_data = item_output_object.get('original_input_data')
    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    appliance_list = np.array(item_input_object.get("item_input_params").get('app_list'))

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')

    app_idx = get_app_idx(appliance_list, app)

    cool_idx = np.where(appliance_list == "cooling")[0][0]
    heat_idx = np.where(appliance_list == "heating")[0][0]
    max_allowed_disagg_change = 0.05

    if (app_cons < (hard_limit - 1)) and (app_cons >= 0):

        pick_cons = min(max_allowed_disagg_change * (disagg_cons[target_days].sum() / Cgbdisagg.WH_IN_1_KWH),
                        ((hard_limit - app_cons) * (np.sum(target_days) / Cgbdisagg.DAYS_IN_MONTH)) * 0.9) * Cgbdisagg.WH_IN_1_KWH

        additional_cons = np.zeros_like(processed_input_data[target_days])

        # picking required consumption from HVAC cons
        # this consumption is picked in a box type signature
        # A limited amount of box type consumption is added from HVAC to stat app
        # to maintain consistency at billing cycle level output

        cons_limit = 500/samples_per_hour

        if np.any(final_tou_consumption[app_idx] > 0):
            cons_limit = np.percentile(final_tou_consumption[app_idx][final_tou_consumption[app_idx] > 0], 80)

        additional_cons[:, stat_tou] = disagg_cons[target_days][:, stat_tou]

        additional_cons[disagg_cons[target_days] < cons_limit] = 0

        if (not (np.max(disagg_cons[target_days][:, stat_tou]) == 0)) and (np.sum(np.nan_to_num(additional_cons)) > 0):
            seed = RandomState(random_gen_config.seed_value)

            # checking consumption that can be picked up from hvac

            additional_cons = add_prepare_box_type_additional_cons_points(pick_cons, final_tou_consumption,
                                                                          additional_cons, vacation, target_days,
                                                                          app_idx, seed)
            target = np.fmax(0, np.minimum(
                processed_input_data[target_days] - final_tou_consumption[app_idx, target_days, :],
                np.fmax(0, np.minimum(disagg_cons[target_days], additional_cons))))

            # adding the consumption into stat app and removing from hvac

            final_tou_consumption[app_idx, target_days, :] = final_tou_consumption[app_idx, target_days, :] + target
            final_tou_consumption[cool_idx, target_days, :] = np.fmax(0, final_tou_consumption[cool_idx, target_days, :] - target)
            final_tou_consumption[heat_idx, target_days, :] = np.fmax(0, final_tou_consumption[heat_idx, target_days, :] - target)
            hvac_cons_val[target_days] = hvac_cons_val[target_days] + np.fmax(0, np.minimum(disagg_cons[target_days], additional_cons))
            disagg_cons[target_days] = disagg_cons[target_days] - np.minimum(disagg_cons[target_days], additional_cons)

            app_cons = get_app_cons_in_target_bc(final_tou_consumption, vacation, target_days, app_idx)

    if (app_cons < hard_limit) and (app_cons >= 0):

        # picking required consumption from residual data

        additional_cons = np.zeros_like(processed_input_data[target_days])

        active_hours = np.ones_like(stat_tou)
        active_hours[inactive_hours] = 0

        other_cons_arr = np.fmax(0, processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0))
        box_cons_others = other_cons_arr

        additional_cons[:, active_hours] = np.fmin(max_cons, box_cons_others[target_days][:, active_hours]) / np.max(
            np.fmin(max_cons, box_cons_others[target_days][:, active_hours]))

        additional_cons = add_randomness(additional_cons)
        additional_cons[vacation[target_days]] = 0

        additional_cons = np.nan_to_num(additional_cons)

        if np.sum(np.nan_to_num(additional_cons)) > 0:
            additional_cons = prepare_additional_cons_points_to_satisfy_min_cons(additional_cons, hard_limit,
                                                                                 target_days, app_cons)

            other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

            final_tou_consumption[app_idx, target_days, :] = \
                final_tou_consumption[app_idx, target_days, :] + np.fmax(0, np.minimum(other_cons_arr[target_days, :] * 0.99, additional_cons).astype(int))

            final_tou_consumption[app_idx, target_days, :] = np.minimum(final_tou_consumption[app_idx, target_days, :], processed_input_data[target_days])

            final_tou_consumption[app_idx, target_days, :] = np.fmin(final_tou_consumption[app_idx, target_days, :], max_cons)

            app_cons = get_app_cons_in_target_bc(final_tou_consumption, vacation, target_days, app_idx)

    if (app_cons < (hard_limit - 1)) and (app_cons >= 0):

        # picking required consumption from HVAC cons
        # A limited amount of monthly consumption is added from HVAC to stat app
        # to maintain consistency at billing cycle level output

        additional_cons = np.zeros_like(processed_input_data[target_days])

        pick_cons = min(max_allowed_disagg_change * (disagg_cons[target_days].sum() / Cgbdisagg.WH_IN_1_KWH),
                        ((hard_limit - app_cons) * (np.sum(target_days) / Cgbdisagg.DAYS_IN_MONTH)) * 0.9) * Cgbdisagg.WH_IN_1_KWH
        additional_cons[:, stat_tou] = np.fmin(max_cons, disagg_cons[target_days][:, stat_tou]) / np.max(
            np.fmin(max_cons, disagg_cons[target_days][:, stat_tou]))

        if not (np.max(disagg_cons[target_days][:, stat_tou]) == 0):
            additional_cons = prepare_ts_level_additional_cons_points(additional_cons, vacation, pick_cons, target_days)

            target = np.fmax(0, np.minimum(
                processed_input_data[target_days] - final_tou_consumption[app_idx, target_days, :],
                np.fmax(0, np.minimum(disagg_cons[target_days], additional_cons))))

            # adding the consumption into stat app and removing from hvac

            final_tou_consumption[app_idx, target_days, :] = final_tou_consumption[app_idx, target_days, :] + target
            final_tou_consumption[cool_idx, target_days, :] = np.fmax(0, final_tou_consumption[cool_idx, target_days, :] - target)
            final_tou_consumption[heat_idx, target_days, :] = np.fmax(0, final_tou_consumption[heat_idx, target_days, :] - target)
            hvac_cons_val[target_days] = hvac_cons_val[target_days] + np.fmax(0, np.minimum(disagg_cons[target_days], additional_cons))
            disagg_cons[target_days] = disagg_cons[target_days] - np.minimum(disagg_cons[target_days], additional_cons)

            app_cons = get_app_cons_in_target_bc(final_tou_consumption, vacation, target_days, app_idx)

    return final_tou_consumption, app_cons, disagg_cons, hvac_cons_val
