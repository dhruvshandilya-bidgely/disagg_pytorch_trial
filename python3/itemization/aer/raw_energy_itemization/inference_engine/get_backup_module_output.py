
"""
Author - Nisha Agarwal
Date - 4th April 2022
calculate pp/ev cons in cases where pp/ev is not detected but shall be forcefully added based on pilot config
"""

# Import python packages

import copy
import numpy as np
import pandas as pd

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config

from python3.itemization.aer.functions.get_hybrid_v2_generic_config import get_hybrid_v2_generic_config

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_step3_app_estimation_config


def add_ev_pp_backup_cons(item_input_object, item_output_object, residual, min_cons_val, mid_cons_val, max_cons_val, logger):

    """
    This function adds step 3 consumption for EV/PP/WH
    step 3 consumption means that the consumption is added when to app profile count is non zero
    and user should be provided non zero output based on hybrid config
    this step 3 consumption is added when both disagg and hybrid does not provide ts level appliance disagg
    this consumption is added only based on monthly level consumption and is considered as statistical output

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        residual                  (np.ndarray)    : disagg residual
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        logger                    (logger)        : logger object

    Returns:
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
    """

    # fetching information that for what all appliances step 3 consumption needs to be addded

    appliances, avg_cons, min_cons, max_cons, residual, min_cons_for_step3_app = \
        prepare_list_of_target_app(residual, item_input_object, item_output_object, mid_cons_val, logger)

    if len(appliances) == 0:
        return min_cons_val, mid_cons_val, max_cons_val

    input_data = item_output_object.get('original_input_data')

    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    residual = np.fmax(0, residual)

    # fetching disagg information and billing cycle data

    heating_ts_level_cons = mid_cons_val[get_index('heating')]
    cooling_ts_level_cons = mid_cons_val[get_index('cooling')]

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    unique_bc = unique_bc[counts > 5]
    monthly_residual = np.zeros(len(unique_bc))

    date_list = item_output_object.get("date_list")
    month_list = pd.DatetimeIndex(date_list).month.values

    months_restriction = np.ones((len(appliances), 12))

    samples = int(input_data.shape[1]/Cgbdisagg.HRS_IN_DAY)

    appliance_cons = np.zeros((len(appliances), len(unique_bc)))

    cooling_bc_cons = np.zeros_like(unique_bc)
    heating_bc_cons = np.zeros_like(unique_bc)

    avg_cons = np.array(avg_cons)
    avg_cons = avg_cons * (np.median(counts) / 30)

    max_cons = np.array(max_cons)
    max_cons = max_cons * (np.median(counts) / 30)

    min_cons = np.array(min_cons)
    min_cons = min_cons * (np.median(counts) / 30)

    # for each billing cycle, calculating the aggregate hvac consumption

    for i in range(len(unique_bc)):

        target_days = bc_list == unique_bc[i]

        monthly_residual[i] = np.sum(residual[target_days])

        valid_months = np.unique(month_list[target_days]) - 1

        valid_appliances = months_restriction[:, valid_months].sum(axis=1) > 0

        cooling_bc_cons[i] = cooling_ts_level_cons[target_days].sum()
        heating_bc_cons[i] = heating_ts_level_cons[target_days].sum()

        #  monthly consumption that can be alloted to each step 3 appliance category
        # based on their individual average consumption and total leftover consumption in a given billing cycle

        if np.sum(valid_appliances):

            avg_cons_list = avg_cons[valid_appliances]

            cons_list = monthly_residual[i] * avg_cons_list / (np.sum(avg_cons_list))

            cons_list = np.fmax(0, cons_list)

            cons_list = np.minimum(cons_list, max_cons[valid_appliances])
            cons_list = np.maximum(cons_list, min_cons[valid_appliances])

            cons_list[np.isin(appliances, ['ref', 'li', 'wh'])] = \
                avg_cons[np.isin(appliances, ['ref', 'li', 'wh'])]

            cons_list[np.isin(appliances, ['cooling', 'heating'])] = \
                np.fmax(cons_list[np.isin(appliances, ['cooling', 'heating'])],
                        monthly_residual[i] - cons_list[np.logical_not(np.isin(appliances, ['cooling', 'heating']))].sum())

            appliance_cons[valid_appliances, i] = cons_list

    # Capping the billing cycle level consumption of each appliance to avoid high outlier consumption

    cons_list = appliance_cons

    config = get_step3_app_estimation_config()

    perc_thres = config.get('perc_thres')

    for i in range(len(appliances)):
        if np.sum(cons_list[i]) > 0:
            cons_list[i] = np.fmin(cons_list[i], np.percentile(cons_list[i][cons_list[i] > 0], perc_thres))

    step3_app_cons = np.zeros((len(appliances), len(cooling_ts_level_cons), len(cooling_ts_level_cons[0])))

    # In each billing cycle, if the consumption is less than certain threshold,
    # small fraction of consumption is taken from hvac consumption
    # keeping a check on max consumption that can be taken from hvac

    for i in range(len(appliances)):
        step3_app_cons, cooling_ts_level_cons, heating_ts_level_cons = \
            pick_cons_from_hvac(cons_list, i, item_input_object, cooling_bc_cons, heating_bc_cons,
                                step3_app_cons, cooling_ts_level_cons, heating_ts_level_cons)

    for i in range(len(appliances)):
        for j in range(len(unique_bc)):

            target_days = bc_list == unique_bc[j]

            if residual[target_days].sum() > 0:
                step3_app_cons[i][target_days] = step3_app_cons[i][target_days] + \
                                                             residual[target_days] * cons_list[i, j] / residual[target_days].sum()

        step3_app_cons[i][input_data < min_cons_for_step3_app[i]/samples] = 0

    # checking consistency in billing cycle level output of step 3 addition by comparing adjacent billing cycles
    step3_app_cons = np.fmax(0, step3_app_cons)

    for i, app in enumerate(appliances):
        step3_app_cons = \
            maintain_cons_in_step3_output(app, step3_app_cons, bc_list, appliances,
                                          copy.deepcopy(vacation_days), months_restriction, month_list, input_data, min_cons_for_step3_app[i])

    for app in appliances:
        step3_app_cons = remove_outlier_cons(appliances, app, step3_app_cons, copy.deepcopy(vacation_days), item_input_object)

    # month blocking
    mid_cons_val[get_index('cooling')] = cooling_ts_level_cons
    mid_cons_val[get_index('heating')] = heating_ts_level_cons

    for i in range(len(appliances)):
        step3_app_cons[i][vacation_days] = 0

        step3_app_cons[i][input_data < min_cons_for_step3_app[i] / samples] = 0

        min_cons_val[get_index(appliances[i])] = step3_app_cons[i]
        mid_cons_val[get_index(appliances[i])] = step3_app_cons[i]
        max_cons_val[get_index(appliances[i])] = step3_app_cons[i]

    if np.sum(step3_app_cons[0]) > 0:
        item_input_object["item_input_params"]["backup_ev"] = 1

    item_input_object["item_input_params"]["backup_app"] = appliances

    return min_cons_val, mid_cons_val, max_cons_val


def pick_cons_from_hvac(cons_list, app_idx, item_input_object, cooling_bc_cons, heating_bc_cons,
                        step3_app_cons, cooling_ts_level_cons, heating_ts_level_cons):

    """
    This function maintains consistency in step 3 addition by adding a fraction of hvac cons into these app cons

    Parameters:
        cons_list                 (np.ndarray)    : BC level consumption of target step 3 app
        app_idx                   (int)           : app index of target step 3 app
        item_input_object         (dict)          : Dict containing all hybrid inputs
        cooling_bc_cons           (np.ndarray)    : BC level consumption of cooling
        heating_bc_cons           (np.ndarray)    : BC level consumption of heating
        step3_app_cons            (np.ndarray)    : TS level consumption of step 3 app
        cooling_ts_level_cons     (np.ndarray)    : TS level consumption of cooling
        heating_ts_level_cons     (np.ndarray)    : TS level consumption of heating

    Returns:
        step3_app_cons            (np.ndarray)    : TS level consumption of step 3 app
        cooling_ts_level_cons     (np.ndarray)    : TS level consumption of cooling
        heating_ts_level_cons     (np.ndarray)    : TS level consumption of heating
    """

    config = get_step3_app_estimation_config()

    perc_thres = config.get('perc_thres')
    frac_thres = config.get('frac_thres')

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    unique_bc = unique_bc[counts > 5]

    if np.sum(cons_list[app_idx]) > 0:

        cooling_picked_cons = np.zeros_like(unique_bc)
        heating_picked_cons = np.zeros_like(unique_bc)

        min_step_3_app_cons = np.percentile(cons_list[app_idx], perc_thres)

        # picking consumption from cooling

        valid_bc = cons_list[app_idx] < min_step_3_app_cons
        picked_cons = np.minimum(min_step_3_app_cons - cons_list[app_idx], cooling_bc_cons * frac_thres)
        picked_cons = np.fmax(0, picked_cons)
        picked_cons[np.logical_not(valid_bc)] = 0
        cooling_picked_cons = cooling_picked_cons + picked_cons

        # picking consumption from heating

        valid_bc = cons_list[app_idx] < min_step_3_app_cons
        picked_cons = np.minimum(min_step_3_app_cons - cons_list[app_idx], heating_bc_cons * frac_thres)
        picked_cons = np.fmax(0, picked_cons)
        picked_cons[np.logical_not(valid_bc)] = 0

        heating_picked_cons = heating_picked_cons + picked_cons

        # removing consumption from hvac consumption

        for j in range(len(unique_bc)):

            target_days = bc_list == unique_bc[j]

            picked_hvac_cons = (cooling_picked_cons[j] / cooling_bc_cons[j]) * cooling_ts_level_cons[target_days]

            if cooling_bc_cons[j] > 0:
                step3_app_cons[app_idx][target_days] = step3_app_cons[app_idx][target_days] + picked_hvac_cons
                cooling_ts_level_cons[target_days] = cooling_ts_level_cons[target_days] - picked_hvac_cons

            picked_hvac_cons = (heating_picked_cons[j] / heating_bc_cons[j]) * heating_ts_level_cons[target_days]

            if heating_bc_cons[j] > 0:
                step3_app_cons[app_idx][target_days] = step3_app_cons[app_idx][target_days] + picked_hvac_cons
                heating_ts_level_cons[target_days] = heating_ts_level_cons[target_days] - picked_hvac_cons

    return step3_app_cons, cooling_ts_level_cons, heating_ts_level_cons


def remove_outlier_cons(appliances, app, step3_app_cons, vacation_days, item_input_object):

    """
    This function maintains consistency in step 3 addition by capping low/high consumption billing cycles

    Parameters:
        appliances                (list)          : list of all appliances
        app                       (str)           : target appliance
        step3_app_cons            (np.ndarray)    : TS level consumption of step 3 app
        vacation_days             (np.ndarray)    : vacation data
        item_input_object         (dict)          : Dict containing all hybrid inputs

    Returns:
        step3_app_cons            (np.ndarray)    : TS level consumption of step 3 app
    """

    # preparing billing cycle data

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    app_idx = np.where(np.array(appliances) == app)[0][0]

    monthly_cons = np.zeros(len(unique_bc))

    if app in ['ref', 'pp']:
        vacation_days[:] = 0

    if len(unique_bc[unique_bc > 0]) < 4:
        return step3_app_cons

    if np.sum(step3_app_cons[app_idx]) == 0:
        return step3_app_cons

    # calculate appliance consumption at bill cycle level

    for i in range(len(unique_bc)):

        target_days = bc_list == unique_bc[i]

        if np.sum(target_days) <= 8:
            continue

        monthly_cons[i] = np.sum(step3_app_cons[app_idx][target_days])

        factor = 1 - (vacation_days[target_days].sum() / np.sum(target_days))

        monthly_cons[i] = monthly_cons[i] * (Cgbdisagg.DAYS_IN_MONTH / np.sum(target_days)) / factor

        if factor == 0:
            monthly_cons[i] = 0

    # calculates max bc level consumption for all appliances

    if np.all(monthly_cons <= 0):
        median_cons = 0

    else:
        median_cons = np.percentile(monthly_cons[monthly_cons > 0], 40)

    if app in ['cooling', 'heating']:
        median_cons = np.percentile(monthly_cons[monthly_cons > 0], 80)

    high_cons_month = monthly_cons > median_cons

    if np.any(high_cons_month):
        high_cons_month_list = np.where(high_cons_month)[0]

        for i in high_cons_month_list:
            target_days = bc_list == unique_bc[i]
            factor = (monthly_cons[i] / median_cons)
            step3_app_cons[app_idx][target_days] = \
                step3_app_cons[app_idx][target_days] / factor

    return step3_app_cons


def prepare_list_of_target_app(residual, item_input_object, item_output_object, mid_cons_val, logger):

    """
    This function maintains consistency in step 3 addition by capping low/high consumption billing cycles

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        logger                    (logger)        : logger object

    Returns:
        appliances                (list)          : list of all appliances
        avg_cons                  (list)          : list of avg values of all appliances
        min_cons                  (list)          : list of min values of all appliances
        max_cons                  (list)          : list of max values of all appliances

    """

    # checking if EV output should be added based on step 3 addition
    # if EV was added based on step 1/2 addition in previous runs, step 3 consumption is not added

    give_ev_to_app_profile_users = 1

    if item_input_object.get("config").get('disagg_mode') in ['incremental', 'mtd'] and \
            item_input_object.get("item_input_params").get('ev_hsm') is not None \
            and item_input_object.get("item_input_params").get('ev_hsm').get('item_type') is not None:
        ev_hsm = item_input_object.get("item_input_params").get('ev_hsm').get('item_type')

        if isinstance(ev_hsm, list):
            type = ev_hsm[0]
        else:
            type = ev_hsm

        if type in [1, 2, 3]:
            give_ev_to_app_profile_users = 0

    logger.info('EV step 3 additional flag based on HSM data| %s', bool(give_ev_to_app_profile_users))

    app_profile = get_profile_count(item_input_object, 'ev')

    appliances = []

    avg_cons = np.array([])
    min_cons = np.array([])
    max_cons = np.array([])

    config = get_step3_app_estimation_config()

    pilot_config = item_input_object.get('pilot_level_config')

    ev_range = config.get('ev_range')
    pp_range = config.get('pp_range')

    min_cons_for_step3_app = np.array([])

    take_from_disagg_flag = pilot_config.get('ev_config').get('bounds').get('take_from_disagg')
    take_from_disagg_flag = (take_from_disagg_flag == 0) or \
                            (take_from_disagg_flag == 2 and (np.sum(mid_cons_val[get_index('ev')]) == 0) and give_ev_to_app_profile_users)

    if (app_profile > 0) and take_from_disagg_flag:
        appliances = ['ev']

        avg_cons = np.array([ev_range[1]])
        min_cons = np.array([ev_range[0]])
        max_cons = np.array([ev_range[2]])
        min_cons_for_step3_app = np.append(min_cons_for_step3_app, 500)

        logger.info('EV step 3 additional final flag | %s', True)
    else:
        logger.info('EV step 3 additional final flag | %s', False)

    # checking if PP output should be added based on step 3 addition

    app_profile = get_profile_count(item_input_object, 'pp')

    take_from_disagg_flag = pilot_config.get('pp_config').get('bounds').get('take_from_disagg')
    take_from_disagg_flag = (take_from_disagg_flag == 0) or (take_from_disagg_flag == 2 and (np.sum(mid_cons_val[get_index('pp')]) == 0))

    if app_profile > 0 and take_from_disagg_flag:
        appliances = np.append(appliances, 'pp')

        avg_cons = np.append(avg_cons, pp_range[1])
        min_cons = np.append(min_cons, pp_range[0])
        max_cons = np.append(max_cons, pp_range[2])

        min_cons_for_step3_app = np.append(min_cons_for_step3_app, 200)

        logger.info('PP step 3 additional final flag | %s', True)
    else:
        logger.info('PP step 3 additional final flag | %s', False)

    appliances, avg_cons, min_cons, max_cons, min_cons_for_step3_app = \
        prepare_list_of_wh_app(residual, item_input_object, appliances, avg_cons, min_cons, max_cons, mid_cons_val, min_cons_for_step3_app, logger)

    appliances, avg_cons, min_cons, max_cons, min_cons_for_step3_app = \
        prepare_list_of_stat_app(residual, item_input_object, appliances, avg_cons, min_cons, max_cons, mid_cons_val, min_cons_for_step3_app, logger)

    appliances, avg_cons, min_cons, max_cons, min_cons_for_step3_app = \
        prepare_list_of_hvac_app(residual, item_input_object, appliances, avg_cons, min_cons, max_cons, mid_cons_val, min_cons_for_step3_app, logger)

    for app in appliances:
        residual = residual + mid_cons_val[get_index(app)]

    return appliances, avg_cons, min_cons, max_cons, residual, min_cons_for_step3_app


def prepare_list_of_wh_app(residual, item_input_object, appliances, avg_cons, min_cons, max_cons, mid_cons_val, min_cons_for_step3_app, logger):
    """
    This function maintains consistency in step 3 addition by capping low/high consumption billing cycles

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        logger                    (logger)        : logger object

    Returns:
        appliances                (list)          : list of all appliances
        avg_cons                  (list)          : list of avg values of all appliances
        min_cons                  (list)          : list of min values of all appliances
        max_cons                  (list)          : list of max values of all appliances

    """

    # checking if EV output should be added based on step 3 addition
    # if EV was added based on step 1/2 addition in previous runs, step 3 consumption is not added

    config = get_step3_app_estimation_config()

    pilot_config = item_input_object.get('pilot_level_config')

    wh_range = config.get('wh_range')

    # checking if WH output should be added based on step 3 addition

    app_profile = get_profile_count(item_input_object, 'wh', wh_output=np.sum(mid_cons_val[get_index('wh')]) > 0)

    cov = int(np.nan_to_num(pilot_config.get('wh_config').get('coverage')))
    fuel_type = str(np.nan_to_num(pilot_config.get('wh_config').get('type')))

    take_from_disagg_flag = pilot_config.get('wh_config').get('bounds').get('take_from_disagg')
    add_wh = (take_from_disagg_flag == 0) and (cov > 50) and (fuel_type == 'ELECTRIC')
    add_wh = add_wh or ((take_from_disagg_flag == 2) and (cov > 95) and (np.sum(mid_cons_val[get_index('wh')]) == 0) and (fuel_type == 'ELECTRIC'))

    if (np.sum(mid_cons_val[get_index('wh')]) == 0 and app_profile and (
            pilot_config.get('wh_config').get('bounds').get('take_from_disagg') in [0, 2])) \
            or add_wh:
        appliances = np.append(appliances, 'wh')
        avg_cons = np.append(avg_cons, wh_range[1])
        min_cons = np.append(min_cons, wh_range[0])
        max_cons = np.append(max_cons, wh_range[2])
        min_cons_for_step3_app = np.append(min_cons_for_step3_app, 100)
        logger.info('WH step 3 additional final flag | %s', True)
    else:
        logger.info('WH step 3 additional final flag | %s', False)

    return appliances, avg_cons, min_cons, max_cons, min_cons_for_step3_app


def prepare_list_of_stat_app(residual, item_input_object, appliances, avg_cons, min_cons, max_cons, mid_cons_val, min_cons_for_step3_app, logger):
    """
    This function maintains consistency in step 3 addition by capping low/high consumption billing cycles

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        logger                    (logger)        : logger object

    Returns:
        appliances                (list)          : list of all appliances
        avg_cons                  (list)          : list of avg values of all appliances
        min_cons                  (list)          : list of min values of all appliances
        max_cons                  (list)          : list of max values of all appliances

    """

    # checking if EV output should be added based on step 3 addition
    # if EV was added based on step 1/2 addition in previous runs, step 3 consumption is not added

    pilot_config = item_input_object.get('pilot_level_config')

    li_range = np.array([pilot_config.get('li_config').get('bounds').get('min_cons'),
                         pilot_config.get('li_config').get('bounds').get('mid_cons'),
                         pilot_config.get('li_config').get('bounds').get('max_cons')]) * 1000

    ref_range = np.array([pilot_config.get('ref_config').get('bounds').get('min_cons'),
                          pilot_config.get('ref_config').get('bounds').get('mid_cons'),
                          pilot_config.get('ref_config').get('bounds').get('max_cons')]) * 1000

    take_from_disagg_flag = pilot_config.get('ref_config').get('bounds').get('take_from_disagg')
    take_from_disagg_flag = (take_from_disagg_flag == 0) or (take_from_disagg_flag == 2 and (np.sum(mid_cons_val[get_index('ref')]) == 0))

    count = 1
    count_based_scaling_factor = 1
    app_profile = item_input_object.get('app_profile').get('ref')

    if (app_profile is not None) and (app_profile.get('number') is not None):
        logger.info('Ref app prof present | ')
        count = app_profile.get('number')

    if count == 2:
        count_based_scaling_factor = 1.75

    if count == 3:
        count_based_scaling_factor = 2.5

    if count >= 4:
        count_based_scaling_factor = 3

    if take_from_disagg_flag:
        appliances = np.append(appliances, 'ref')
        avg_cons = np.append(avg_cons, ref_range[1] * count_based_scaling_factor)
        min_cons = np.append(min_cons, ref_range[0])
        max_cons = np.append(max_cons, ref_range[2])
        min_cons_for_step3_app = np.append(min_cons_for_step3_app, 25)

        logger.info('ref step 3 additional final flag | %s', True)
    else:
        logger.info('ref step 3 additional final flag | %s', False)

    take_from_disagg_flag = pilot_config.get('li_config').get('bounds').get('take_from_disagg')
    take_from_disagg_flag = (take_from_disagg_flag == 0) or (take_from_disagg_flag == 2 and (np.sum(mid_cons_val[get_index('li')]) == 0))

    li_config = get_inf_config().get("li")

    efficiency = '1'

    app_profile = item_input_object.get('app_profile').get('li')

    if (app_profile is not None) and (app_profile.get('size') is not None):
        logger.info('Li app prof present | ')
        efficiency = app_profile.get('size')

    efficiency_based_scaling_factor = 1

    if efficiency == '3':
        logger.info("Scaling lighting output, since user has all the li as efficient")
        efficiency_based_scaling_factor = li_config.get('scaling_factor_3')

    if efficiency == '2':
        logger.info("Scaling lighting output, since user has most of the li as efficient")
        efficiency_based_scaling_factor = li_config.get('scaling_factor_2')

    if efficiency == '0':
        logger.info("Scaling lighting output, since user has all the li as inefficient")
        efficiency_based_scaling_factor = li_config.get('scaling_factor_0')

    if take_from_disagg_flag:
        appliances = np.append(appliances, 'li')
        avg_cons = np.append(avg_cons, li_range[1] * efficiency_based_scaling_factor)
        min_cons = np.append(min_cons, li_range[0])
        max_cons = np.append(max_cons, li_range[2])
        min_cons_for_step3_app = np.append(min_cons_for_step3_app, 25)
        logger.info('li step 3 additional final flag | %s', True)
    else:
        logger.info('li step 3 additional final flag | %s', False)

    return appliances, avg_cons, min_cons, max_cons, min_cons_for_step3_app


def prepare_list_of_hvac_app(residual, item_input_object, appliances, avg_cons, min_cons, max_cons, mid_cons_val, min_cons_for_step3_app, logger):
    """
    This function maintains consistency in step 3 addition by capping low/high consumption billing cycles

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        logger                    (logger)        : logger object

    Returns:
        appliances                (list)          : list of all appliances
        avg_cons                  (list)          : list of avg values of all appliances
        min_cons                  (list)          : list of min values of all appliances
        max_cons                  (list)          : list of max values of all appliances

    """

    # checking if EV output should be added based on step 3 addition
    # if EV was added based on step 1/2 addition in previous runs, step 3 consumption is not added

    pilot_config = item_input_object.get('pilot_level_config')

    cooling_range = np.array([pilot_config.get('cooling_config').get('bounds').get('min_cons'),
                              pilot_config.get('cooling_config').get('bounds').get('mid_cons'),
                              pilot_config.get('cooling_config').get('bounds').get('max_cons')]) * 1000

    heating_range = np.array([pilot_config.get('heating_config').get('bounds').get('min_cons'),
                              pilot_config.get('heating_config').get('bounds').get('mid_cons'),
                              pilot_config.get('heating_config').get('bounds').get('max_cons')]) * 1000

    cov = int(np.nan_to_num(pilot_config.get('cooling_config').get('coverage')))
    take_from_disagg_flag = pilot_config.get('cooling_config').get('bounds').get('take_from_disagg')
    take_from_disagg_flag = (take_from_disagg_flag == 0) or (take_from_disagg_flag == 2 and (np.sum(mid_cons_val[get_index('cooling')]) == 0))

    if (take_from_disagg_flag and cov > 50):
        appliances = np.append(appliances, 'cooling')

        cooling_range[1] = max(cooling_range[1], 50000)
        avg_cons = np.append(avg_cons, cooling_range[1])
        min_cons = np.append(min_cons, cooling_range[0])
        max_cons = np.append(max_cons, cooling_range[2])
        min_cons_for_step3_app = np.append(min_cons_for_step3_app, 50)

        logger.info('cooling step 3 additional final flag | %s', True)
    else:
        logger.info('cooling step 3 additional final flag | %s', False)

    cov = int(np.nan_to_num(pilot_config.get('heating_config').get('coverage')))
    take_from_disagg_flag = pilot_config.get('heating_config').get('bounds').get('take_from_disagg')
    take_from_disagg_flag = (take_from_disagg_flag == 0) or (take_from_disagg_flag == 2 and (np.sum(mid_cons_val[get_index('heating')]) == 0))

    if take_from_disagg_flag and cov > 50:
        appliances = np.append(appliances, 'heating')
        heating_range[1] = max(heating_range[1], 50000)

        avg_cons = np.append(avg_cons, heating_range[1])
        min_cons = np.append(min_cons, heating_range[0])
        max_cons = np.append(max_cons, heating_range[2])
        min_cons_for_step3_app = np.append(min_cons_for_step3_app, 50)

        logger.info('heating step 3 additional final flag | %s', True)
    else:
        logger.info('heating step 3 additional final flag | %s', False)

    return appliances, avg_cons, min_cons, max_cons, min_cons_for_step3_app


def get_profile_count(item_input_object, app, wh_output=0):

    """
    This function fetches app profile data

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        app                       (str)           : target appliance
        wh_output                 (bool)          : tag that represents whether wh output is 0 in disagg/hybrid

    Returns:
        app_profile               (int)           : count of appliance
    """

    app_profile = item_input_object.get("app_profile").get(app)

    if app_profile is not None:
        app_profile = app_profile.get("number", 0)
    else:
        app_profile = 0

    if app == 'wh':
        app_profile = item_input_object.get("app_profile").get('wh')
        type = "ELECTRIC"
        if app_profile is not None:
            type = app_profile.get("type", 'ELECTRIC')
            app_profile = app_profile.get("number", 0)
        else:
            app_profile = 0

        if type in ['OTHERS', 'Others', 'Other', 'OTHER', 'others', 'COMBINED'] and wh_output == 0:
            app_profile = 0

        if type in ["SOLAR", "PROPANE", "GAS", "Gas", "SOLID_FUEL", "SOLID_FEUL", "OIL", "Oil", "WOOD", "Wood"]:
            app_profile = 0

    return app_profile


def maintain_cons_in_step3_output(app, step3_app_cons, bc_list, appliances, vacation_days, months_restriction, month_list, input_data, min_cons_for_step3_app):

    """
    This function maintains consistency in step 3 addition by capping low/high consumption billing cycles

    Parameters:
        app                       (str)           : target appliance
        step3_app_cons            (np.ndarray)    : TS level consumption of step 3 app
        bc_list                   (np.ndarray)    : billing cycle data
        appliances                (list)          : list of all appliances
        vacation_days             (np.ndarray)    : vacation data
        months_restriction        (np.ndarray)    : list of months where output is restricted for an appliance
        month_list                (np.ndarray)    : month data
        input_data                (np.ndarray)    : user input data

    Returns:
        step3_app_cons            (np.ndarray)    : TS level consumption of step 3 app
    """

    config = get_step3_app_estimation_config()

    min_scal_fac = config.get('min_scal_fac')
    max_scal_fac = config.get('max_scal_fac')

    unique_bc, bc_size = np.unique(bc_list, return_counts=True)

    samples = input_data.shape[1]/Cgbdisagg.HRS_IN_DAY

    min_bc_required_for_consistency_check = get_hybrid_v2_generic_config().get('min_bc_required_for_consistency_check')

    monthly_cons = np.zeros(len(unique_bc))

    app_idx = np.where(np.array(appliances) == app)[0][0]

    app_cons = step3_app_cons[app_idx]

    if app in ['ref', 'pp']:
        vacation_days[:] = 0

    # preparing monthly consumption of the appliance, scaled based on vacation

    for i in range(len(unique_bc)):

        if np.sum(np.sum(bc_list == unique_bc[i])) <= 7:
            continue

        target_days = bc_list == unique_bc[i]

        monthly_cons[i] = np.sum(app_cons[target_days])

        factor = 1 - (vacation_days[target_days].sum() / np.sum(target_days))

        monthly_cons[i] = monthly_cons[i] * (Cgbdisagg.DAYS_IN_MONTH / np.sum(target_days)) / factor

        monthly_cons[i] = monthly_cons[i] * (1 - (factor == 0))

    for i in range(0, len(unique_bc)):

        target_days = bc_list == unique_bc[i]

        # if appliance consumption in a billing cycle is inconsistent with neighbouring billing cycles,
        # consumption values are scaled to dec/inc consumption

        if len(unique_bc) < min_bc_required_for_consistency_check:
            break

        if i == 0:

            if np.any(monthly_cons[i:(i+2)] == 0):
                continue

            diff = (monthly_cons[i]) / (monthly_cons[i+1])

            diff = diff*(diff >= 1.1) + (diff < 1.1)

            diff = max(min_scal_fac, diff)
            diff = min(diff, max_scal_fac)
            step3_app_cons[app_idx][target_days] = step3_app_cons[app_idx][target_days] / diff

        elif i == len(unique_bc)-1:

            if np.any(monthly_cons[i-1:(i+1)] == 0):
                continue

            diff = (monthly_cons[i]) / (monthly_cons[i-1])

            diff = diff * (diff >= 1.1) + (diff < 1.1)

            diff = max(min_scal_fac, diff)
            diff = min(diff, max_scal_fac)
            step3_app_cons[app_idx][target_days] = step3_app_cons[app_idx][target_days] / diff

        elif not (np.any(monthly_cons[i - 1:(i + 2)] == 0)):
            diff = (monthly_cons[i]) / ((monthly_cons[i + 1] + monthly_cons[i - 1]) / 2)

            diff = diff*(diff >= 1.1) + (diff < 1.1)

            diff = max(min_scal_fac, diff)
            diff = min(diff, max_scal_fac)

            step3_app_cons[app_idx][target_days] = step3_app_cons[app_idx][target_days] / diff

    block_month = np.where(months_restriction[app_idx] == 0)[0]
    step3_app_cons[app_idx][np.isin(month_list, block_month)] = 0

    step3_app_cons[app_idx][input_data < min_cons_for_step3_app / samples] = 0

    step3_app_cons = np.fmax(0, step3_app_cons)

    return step3_app_cons


def get_index(app_name):

    """
    fetch index of appliance from the list of all target appliances

    Parameters:
        app_name                    (str)           : target Appliance name

    """

    appliance_list = np.array(["ao", "ev", "cooling", "heating", 'li', "pp", "ref", "va1", "va2", "wh", "ld", "ent", "cook"])

    return int(np.where(appliance_list == app_name)[0][0])

