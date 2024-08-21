
"""
Author - Nisha Agarwal
Date - 4th April 2021
Utils functions for calculating 100% itemization module
"""

# Import python packages

import copy
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.functions.get_hybrid_v2_generic_config import get_hybrid_v2_generic_config


def stat_app_consistency_check(final_tou_consumption, app_list, bc_list, vacation, logger):

    """
    Maintain bc level consistency for stat appliances

    Parameters:
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)
        app_list                    (list)          : list of appliances
        bc_list                     (np.ndarray)    : list of bc start ts of all target days
        vacation                    (np.ndarray)    : vacation info
        logger                      (logger)        : logger object

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    unique_bc = np.unique(bc_list)

    median_cons = 0

    stat_app_perc_cap = 40

    buffer_for_high_cons_month = 1.1

    for app in ['cook', 'ent', 'ld']:

        app_idx = np.where(app_list == app)[0][0]

        # calculate appliance consumption at bill cycle level

        if np.sum(final_tou_consumption[app_idx]) == 0:
            continue

        monthly_cons, final_tou_consumption = \
            get_app_monthly_cons(final_tou_consumption, vacation, unique_bc, bc_list, app, app_list, final_tou_consumption[app_idx])

        # calculate max bc level consumption and modify consumption in all billing cycles

        maintain_consistency_bool = 0

        if np.sum(monthly_cons > 0) > 0:
            median_cons = np.percentile(monthly_cons[monthly_cons > 0], stat_app_perc_cap)
            maintain_consistency_bool = 1

        high_cons_month = monthly_cons > median_cons * buffer_for_high_cons_month

        logger.info("appliance for which max cons is applied | %s", app)
        logger.info("appliance median consumption | %s", median_cons)

        # reduce output in billing cycles where consumption is overshooting

        if np.any(high_cons_month) and maintain_consistency_bool:
            high_cons_month_list = np.where(high_cons_month)[0]

            for i in high_cons_month_list:
                target_days = bc_list == unique_bc[i]

                factor = (monthly_cons[i] / median_cons) / buffer_for_high_cons_month
                final_tou_consumption[app_idx][target_days] = \
                    final_tou_consumption[app_idx][target_days] / np.fmax(1, factor)

    return final_tou_consumption


def get_app_monthly_cons(final_tou_consumption, vacation, unique_bc, bc_list, app, app_list, app_cons):

    """
    get monthly level consumption of appliances scaled to 30 days and scaled based on non vacation days

    Parameters:
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)
        vacation                    (np.ndarray)    : vacation info
        unique_bc                   (np.ndarray)    : list of unique billing cycles
        bc_list                     (np.ndarray)    : list of bc start ts of all target days
        app                         (str)           : target app
        app_list                    (np.ndarray)    : list of all app
        app_cons                    (np.ndarray)    : cons of target app

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    monthly_cons = np.zeros(len(unique_bc))

    if app == 'others':
        app_idx = -1
    else:
        app_idx = np.where(app_list == app)[0][0]

    for i in range(len(unique_bc)):

        target_days = bc_list == unique_bc[i]

        if np.sum(np.sum(target_days)) <= Cgbdisagg.DAYS_IN_WEEK:
            continue

        monthly_cons[i] = np.sum(app_cons[target_days])

        monthly_cons[i] = monthly_cons[i] * (Cgbdisagg.DAYS_IN_MONTH / np.sum(target_days))

        if app in ['pp', 'ev']:
            final_tou_consumption[app_idx][target_days] = 0
        elif (1 - (np.sum(vacation[target_days]) / np.sum(target_days))) == 0:
            monthly_cons[i] = 0
        else:
            monthly_cons[i] = monthly_cons[i] / (1 - (np.sum(vacation[target_days]) / np.sum(target_days)))

    return monthly_cons, final_tou_consumption


def consistency_check_for_low_cons_app(final_tou_consumption, app_list, bc_list, vacation, logger):

    """
    Maintain bc level consistency for stat appliances

    Parameters:
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)
        app_list                    (list)          : list of appliances
        bc_list                     (np.ndarray)    : list of bc start ts of all target days
        vacation                    (np.ndarray)    : vacation info
        logger                      (logger)        : logger object

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    min_bc_required_for_consistency_check = get_hybrid_v2_generic_config().get('min_bc_required_for_consistency_check')

    unique_bc, bc_size = np.unique(bc_list, return_counts=True)

    unique_bc = unique_bc[bc_size >= min_bc_required_for_consistency_check]

    target_app_list = ['cook', 'ent', 'ld']

    buffer_for_high_cons_month = 1.1

    max_perc_for_cons_check = 50
    min_perc_for_cons_check = 5
    scaling_fac_for_cons_check_perc = 900

    # the percentile used will depend upon the length of billing cycle

    thres = min(max_perc_for_cons_check, max(min_perc_for_cons_check,
                                             int(scaling_fac_for_cons_check_perc / np.median(np.unique(bc_list, return_counts=1)[1]) )))

    final_tou_consumption = np.fmax(0, final_tou_consumption)

    for app in target_app_list:

        app_idx = np.where(app_list == app)[0][0]

        if len(unique_bc[unique_bc > 0]) < 4:
            continue

        if np.sum(final_tou_consumption[app_idx]) == 0:
            continue

        # calculate appliance consumption at bill cycle level
        monthly_cons, final_tou_consumption = get_app_monthly_cons(final_tou_consumption, vacation, unique_bc, bc_list,
                                                                   app, app_list, final_tou_consumption[app_idx])

        if np.sum(monthly_cons > 0) > 0:
            median_cons = np.percentile(monthly_cons[monthly_cons > 0], thres) * buffer_for_high_cons_month
        else:
            median_cons = 0

        high_cons_month = monthly_cons > median_cons

        logger.info("appliance for which max cons is applied | %s", app)
        logger.info("appliance median consumption | %s", median_cons)

        if np.any(high_cons_month):
            high_cons_month_list = np.where(high_cons_month)[0]

            for i in high_cons_month_list:

                target_days = bc_list == unique_bc[i]

                factor = (monthly_cons[i] / median_cons)
                final_tou_consumption[app_idx][target_days] = \
                    final_tou_consumption[app_idx][target_days] / factor

    return final_tou_consumption


def consistency_check_for_all_output_based_on_stat_estimation(input_data, item_input_object, final_tou_consumption, app_list, bc_list, vacation):

    """
    This function ensure max bc level consumption. This function is added to avoid any high consumption
     being added after itemization

    Parameters:
        input_data                  (np.ndarray)    : raw input data
        item_input_object           (dict)          : Dict containing all hybrid inputs
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)
        app_list                    (list)          : list of target appliances
        bc_list                     (np,ndarray)    : list of billing cycles
        vacation                    (np.ndarray)    : list of vacation tags

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    min_bc_required_for_consistency_check = get_hybrid_v2_generic_config().get('min_bc_required_for_consistency_check')

    unique_bc, bc_size = np.unique(bc_list, return_counts=True)

    unique_bc = unique_bc[bc_size >= min_bc_required_for_consistency_check]

    # target appliances are appliances with statistical output

    target_app_list = ['cook', 'ent', 'ld']

    perc_thres = [45, 50, 50]

    if 'pp' in item_input_object["item_input_params"]["backup_app"]:
        target_app_list.append('pp')
        perc_thres.append(35)

    if 'ev' in item_input_object["item_input_params"]["backup_app"]:
        target_app_list.append('ev')
        perc_thres.append(35)

    if 'li' in item_input_object["item_input_params"]["backup_app"]:
        target_app_list.append('li')
        perc_thres.append(50)

    if 'ref' in item_input_object["item_input_params"]["backup_app"]:
        target_app_list.append('ref')
        perc_thres.append(50)

    if 'wh' in item_input_object["item_input_params"]["backup_app"]:
        target_app_list.append('wh')
        perc_thres.append(50)

    final_tou_consumption = np.fmax(0, final_tou_consumption)

    for app in target_app_list:
        final_tou_consumption = \
            consistency_check_for_all_output_based_on_stat_est_for_each_app(app, perc_thres, target_app_list, final_tou_consumption, app_list, bc_list, vacation)

    return final_tou_consumption


def consistency_check_for_all_output_based_on_stat_est_for_each_app(app, perc_thres, target_app_list, final_tou_consumption, app_list, bc_list, vacation):

    """
    This function ensure max bc level consumption. This function is added to avoid any high consumption
     being added after itemization

    Parameters:
        input_data                  (np.ndarray)    : raw input data
        item_input_object           (dict)          : Dict containing all hybrid inputs
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)
        app_list                    (list)          : list of target appliances
        bc_list                     (np,ndarray)    : list of billing cycles
        vacation                    (np.ndarray)    : list of vacation tags

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    unique_bc, bc_size = np.unique(bc_list, return_counts=True)

    # target appliances are appliances with statistical output

    monthly_cons = np.zeros(len(unique_bc))

    app_idx = np.where(app_list == app)[0][0]

    if (np.sum(final_tou_consumption[app_idx]) == 0) or (len(unique_bc[unique_bc > 0]) < 4):
        return final_tou_consumption

    # calculate appliance consumption at bill cycle level

    for i in range(len(unique_bc)):

        target_days = bc_list == unique_bc[i]

        if np.sum(np.sum(target_days)) <= 8:
            continue

        monthly_cons[i] = np.sum(final_tou_consumption[app_idx][target_days])

        monthly_cons[i] = monthly_cons[i] * (Cgbdisagg.DAYS_IN_MONTH/np.sum(target_days))

        if app in ['pp'] and monthly_cons[i] < 10:
            final_tou_consumption[app_idx][target_days] = 0
        elif (1 - (np.sum(vacation[target_days]) / np.sum(target_days))) == 0:
            monthly_cons[i] = 0
        else:
            monthly_cons[i] = monthly_cons[i] / (1 - (np.sum(vacation[target_days]) / np.sum(target_days)))

    # calculates max bc level consumption for all appliances

    if np.sum(monthly_cons > 0) > 0:
        median_cons = np.percentile(monthly_cons[monthly_cons > 0], perc_thres[target_app_list.index(app)])
    else:
        median_cons = 0

    high_cons_month = monthly_cons > median_cons

    if np.any(high_cons_month):
        high_cons_month_list = np.where(high_cons_month)[0]

        for i in high_cons_month_list:
            target_days = bc_list == unique_bc[i]
            factor = (monthly_cons[i] / median_cons)
            final_tou_consumption[app_idx][target_days] = \
                final_tou_consumption[app_idx][target_days] / factor

    return final_tou_consumption


def apply_consistency_in_neighbouring_bcs(item_input_object, final_tou_consumption, vacation, bc_list, all_appliance_list):

    """
    This function ensure max bc level consumption. This function is added to avoid any high consumption
     being added after itemization

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
        vacation                    (np.ndarray)    : list of vacation tags
        bc_list                     (np,ndarray)    : list of billing cycles
        all_appliance_list          (list)          : list of all appliances

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    app_list = ['ld', 'cook', 'ent', 'li', 'ref']

    app_list = np.append(app_list, item_input_object["item_input_params"]["backup_app"])

    app_list = np.unique(app_list)

    input_data = item_input_object.get('item_input_params').get('input_data')[Cgbdisagg.INPUT_CONSUMPTION_IDX]

    vacation2 = np.sum(input_data > 0, axis=1) == 0

    unique_bc, bc_size = np.unique(bc_list, return_counts=True)

    for app in app_list:

        vacation_data = copy.deepcopy(vacation)

        if app in ['ref']:
            vacation_data = copy.deepcopy(vacation2)

        monthly_cons = np.zeros(len(unique_bc))

        app_idx = np.where(all_appliance_list == app)[0][0]

        app_cons = final_tou_consumption[app_idx]

        # calculate monthly consumption

        for i in range(len(unique_bc)):

            target_days = bc_list == unique_bc[i]

            if np.sum(target_days) <= Cgbdisagg.DAYS_IN_WEEK:
                continue

            monthly_cons[i] = np.sum(app_cons[target_days])

            monthly_cons[i] = monthly_cons[i] * (Cgbdisagg.DAYS_IN_MONTH / np.sum(target_days))

            if (1 - (np.sum(vacation_data[target_days]) / np.sum(target_days))) == 0:
                monthly_cons[i] = 0
            else:
                monthly_cons[i] = monthly_cons[i] / (1 - (np.sum(vacation_data[target_days]) / np.sum(target_days)))

        final_tou_consumption = \
            apply_consistency_in_neighbouring_bcs_for_each_app(monthly_cons, final_tou_consumption,
                                                               bc_list, all_appliance_list, app)

    return final_tou_consumption


def apply_consistency_in_neighbouring_bcs_for_each_app(monthly_cons, final_tou_consumption, bc_list,
                                                       all_appliance_list, app):
    """
    This function ensure max bc level consumption. This function is added to avoid any high consumption
     being added after itemization

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
        vacation                    (np.ndarray)    : list of vacation tags
        bc_list                     (np,ndarray)    : list of billing cycles
        all_appliance_list          (list)          : list of all appliances

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    unique_bc, bc_size = np.unique(bc_list, return_counts=True)

    min_bc_required_for_consistency_check = get_hybrid_v2_generic_config().get('min_bc_required_for_consistency_check')

    app_idx = np.where(all_appliance_list == app)[0][0]

    # calculate monthly consumption

    for i in range(0, len(unique_bc)):

        target_days = bc_list == unique_bc[i]

        if len(unique_bc) < min_bc_required_for_consistency_check:
            break

        # compare monthly consumption of neighbouring billing cycles

        if i == 0:

            if np.any(monthly_cons[i:(i + 2)] == 0):
                continue

            cons_diff = (monthly_cons[i]) / (monthly_cons[i + 1])


        elif i == len(unique_bc) - 1:

            if np.any(monthly_cons[i - 1:(i + 1)] == 0):
                continue

            cons_diff = (monthly_cons[i]) / (monthly_cons[i - 1])

        else:
            if np.any(monthly_cons[i - 1:(i + 2)] == 0):
                continue

            cons_diff = (monthly_cons[i]) / ((monthly_cons[i + 1] + monthly_cons[i - 1]) / 2)

        if cons_diff < 1.1:
            cons_diff = 1

        cons_diff = max(1.02, cons_diff)
        cons_diff = min(cons_diff, 1.5)

        final_tou_consumption[app_idx][target_days] = final_tou_consumption[app_idx][target_days] / cons_diff

    return final_tou_consumption
