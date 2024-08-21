

"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Prepare data required in itemization pipeline
"""

# Import python packages

import copy
import numpy as np
import pandas as pd

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config


def get_index(app_name):

    """
    fetch index of appliance from the list of all target appliances

    Parameters:
        app_name                    (str)           : target Appliance name

    """

    appliance_list = np.array(["ao", "ev", "cooling", "heating", 'li', "pp", "ref", "va1", "va2", "wh", "ld", "ent", "cook"])

    return int(np.where(appliance_list == app_name)[0][0])


def adjust_cons(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals, available_cons, conf_vals, hvac_name, month):

    """
    Modify appliance mid/min/max ranges in cases where true disagg appliances are overlapping

    Parameters:
        negative_residual         (np.ndarray)    : tou where disagg residual is negative
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        available_cons            (np.ndarray)    : disagg res
        conf_vals                 (np.ndarray)    : TS level conf vals of all app
        hvac_name                 (str)           : name of hvac appliancee to be adjusted
        month                     (int)           : moth index

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
    """

    pp_idx = get_index('pp')
    ev_idx = get_index('ev')
    wh_idx = get_index('wh')

    wh_cons = mid_cons_vals[wh_idx]
    pp_cons = mid_cons_vals[pp_idx]
    ev_cons = mid_cons_vals[ev_idx]
    hvac_cons = mid_cons_vals[get_index(hvac_name)]

    temp_wh = copy.deepcopy(wh_cons)
    temp_wh[conf_vals[wh_idx] > 1] = 0

    # handle overlap cases of hvac + ev + wh

    if np.sum(hvac_cons[negative_residual]) and np.sum(ev_cons[negative_residual]) and np.sum(
            temp_wh[negative_residual]):
        mid_cons_vals, min_cons_vals, max_cons_vals = \
            adjust_hvac_ev_wh(month, negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals, available_cons,
                              conf_vals, hvac_name)

    # handle overlap cases of hvac + pp + wh

    elif np.sum(hvac_cons[negative_residual]) and np.sum(pp_cons[negative_residual]) and np.sum(
            temp_wh[negative_residual]):
        mid_cons_vals, min_cons_vals, max_cons_vals = \
            adjust_hvac_timed_box(month, negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals,
                                  available_cons, conf_vals, hvac_name, "pp", "wh")

    # handle overlap cases of hvac + pp + ev

    elif np.sum(hvac_cons[negative_residual]) and np.sum(pp_cons[negative_residual]) and np.sum(
            ev_cons[negative_residual]):
        mid_cons_vals, min_cons_vals, max_cons_vals = \
            adjust_hvac_timed_box(month, negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals,
                                  available_cons, conf_vals, hvac_name, "pp", "ev")

    # handle overlap cases of pp + wh

    elif np.sum(pp_cons[negative_residual]) and np.sum(temp_wh[negative_residual]):
        mid_cons_vals, min_cons_vals, max_cons_vals, available_cons = \
            adjust_box_timed(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals,
                             available_cons, conf_vals, "wh", "pp")

    # handle overlap cases of ev + pp

    elif np.sum(pp_cons[negative_residual]) and np.sum(ev_cons[negative_residual]):
        mid_cons_vals, min_cons_vals, max_cons_vals, available_cons = \
            adjust_box_timed(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals,
                             available_cons, conf_vals, "ev", "pp")

    # handle overlap cases of ev + wh

    elif np.sum(ev_cons[negative_residual]) and np.sum(temp_wh[negative_residual]):
        mid_cons_vals, min_cons_vals, max_cons_vals, available_cons = \
            adjust_ev_wh(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals, available_cons, conf_vals, "")

    # handle overlap cases of hvac + wh

    elif np.sum(hvac_cons[negative_residual]) and np.sum(temp_wh[negative_residual]):
        mid_cons_vals, min_cons_vals, max_cons_vals, available_cons = \
            adjust_box_hvac(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals, available_cons,
                            conf_vals, "wh", hvac_name)

    # handle overlap cases of hvac + ev

    elif np.sum(hvac_cons[negative_residual]) and np.sum(ev_cons[negative_residual]):
        mid_cons_vals, min_cons_vals, max_cons_vals, available_cons = \
            adjust_box_hvac(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals, available_cons,
                            conf_vals, "ev", hvac_name)

    # handle overlap cases of hvac + pp

    elif np.sum(hvac_cons[negative_residual]) and np.sum(pp_cons[negative_residual]):
        mid_cons_vals, min_cons_vals, max_cons_vals, available_cons = \
            adjust_hvac_timed(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals, available_cons,
                              conf_vals, hvac_name, 'pp')

    return mid_cons_vals, min_cons_vals, max_cons_vals


def adjust_cons_for_twh_user(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals, available_cons, conf_vals, hvac_name, month):

    """
    Modify appliance mid/min/max ranges in cases where true disagg appliances are overlapping

    Parameters:
        negative_residual         (np.ndarray)    : tou where disagg residual is negative
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        available_cons            (np.ndarray)    : disagg res
        conf_vals                 (np.ndarray)    : TS level conf vals of all app
        hvac_name                 (str)           : name of hvac appliancee to be adjusted
        month                     (int)           : moth index

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
    """

    pp_idx = get_index('pp')
    ev_idx = get_index('ev')
    wh_idx = get_index('wh')

    wh_cons = mid_cons_vals[wh_idx]
    pp_cons = mid_cons_vals[pp_idx]
    ev_cons = mid_cons_vals[ev_idx]
    hvac_cons = mid_cons_vals[get_index(hvac_name)]

    # handle overlap cases of hvac + ev + twh

    if np.sum(hvac_cons[negative_residual]) and np.sum(ev_cons[negative_residual]) and np.sum(
            wh_cons[negative_residual]):
        mid_cons_vals, min_cons_vals, max_cons_vals = \
            adjust_hvac_timed_box(month, negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals,
                                  available_cons, conf_vals, hvac_name, "wh", "ev")

    # handle overlap cases of ev + twh

    elif np.sum(ev_cons[negative_residual]) and np.sum(wh_cons[negative_residual]):
        mid_cons_vals, min_cons_vals, max_cons_vals, available_cons = \
            adjust_box_timed(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals,
                             available_cons, conf_vals, "ev", "wh")

    # handle overlap cases of hvac + twh

    elif np.sum(hvac_cons[negative_residual]) and np.sum(wh_cons[negative_residual]):
        mid_cons_vals, min_cons_vals, max_cons_vals, available_cons = \
            adjust_hvac_timed(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals, available_cons,
                              conf_vals, hvac_name, 'wh')

    # handle overlap cases of hvac + pp

    elif np.sum(hvac_cons[negative_residual]) and np.sum(pp_cons[negative_residual]):
        mid_cons_vals, min_cons_vals, max_cons_vals, available_cons = \
            adjust_hvac_timed(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals, available_cons,
                              conf_vals, hvac_name, 'pp')

    # handle overlap cases of ev + pp

    elif np.sum(pp_cons[negative_residual]) and np.sum(ev_cons[negative_residual]):
        mid_cons_vals, min_cons_vals, max_cons_vals, available_cons = \
            adjust_box_timed(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals,
                             available_cons, conf_vals, "ev", "pp")

    # handle overlap cases of hvac + ev

    elif np.sum(hvac_cons[negative_residual]) and np.sum(ev_cons[negative_residual]):
        mid_cons_vals, min_cons_vals, max_cons_vals, available_cons = \
            adjust_box_hvac(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals, available_cons,
                            conf_vals, "ev", hvac_name)

    return mid_cons_vals, min_cons_vals, max_cons_vals


def adjust_ev_wh(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals, available_cons, conf_vals, hvac_name, hvac_adjust=False):

    """
    Modify appliance mid/min/max ranges in cases where true disagg appliances are overlapping

    Parameters:
        negative_residual         (np.ndarray)    : tou where disagg residual is negative
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        available_cons            (np.ndarray)    : disagg res
        conf_vals                 (np.ndarray)    : TS level conf vals of all app
        hvac_name                 (str)           : name of hvac appliancee to be adjusted
        hvac_adjust               (bool)          : flag that denotes whether hvac needs to be modified incase of negative residual points

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
        available_cons            (np.ndarray)    : updated res
    """

    ev_idx = get_index('ev')
    wh_idx = get_index('wh')

    ev_conf = conf_vals[ev_idx, :, :]
    wh_conf = conf_vals[wh_idx, :, :]

    ev_cons = mid_cons_vals[ev_idx, :, :]
    wh_cons = mid_cons_vals[wh_idx, :, :]

    night_bool = np.zeros_like(ev_conf)
    samples_per_hour = ev_conf.shape[1] / Cgbdisagg.HRS_IN_DAY

    config = get_inf_config(samples_per_hour).get("neg_res_handling")

    night_hours = config.get('night_hours')
    ev_wh_min_conf = config.get('ev_wh_min_conf')
    min_wh_amp = config.get('min_wh_amp')
    wh_conf_offset_for_night_hours = config.get('wh_conf_offset_for_night_hours')
    wh_conf_offset_for_high_cons_points = config.get('wh_conf_offset_for_high_cons_points')
    ev_conf_thres = config.get('ev_conf_thres')
    wh_conf_for_high_conf_ev_points = config.get('wh_conf_for_high_conf_ev_points')

    # increasing confidence score of EV if there is overlap between EV and storage WH during night hours

    night_bool[:, night_hours] = 1

    wh_conf[np.logical_and(ev_conf > 0, night_bool)] = wh_conf[np.logical_and(ev_conf > 0, night_bool)] - wh_conf_offset_for_night_hours

    wh_conf[np.logical_and(ev_conf > 0, mid_cons_vals[wh_idx] > min_wh_amp)] = \
        wh_conf[np.logical_and(ev_conf > 0, mid_cons_vals[wh_idx] > min_wh_amp)] - wh_conf_offset_for_high_cons_points

    wh_conf[ev_conf > ev_conf_thres] = wh_conf_for_high_conf_ev_points

    wh_conf = np.fmax(0, wh_conf)

    ev_cons = ev_cons / np.max(ev_cons)
    wh_cons = wh_cons / np.max(wh_cons)

    conf_diff = wh_conf - ev_conf
    ev_conf[conf_diff > ev_wh_min_conf] = 0
    wh_conf[conf_diff < -ev_wh_min_conf] = 0

    ev_conf = np.nan_to_num(np.multiply(ev_conf, ev_cons))
    wh_conf = np.nan_to_num(np.multiply(wh_conf, wh_cons))

    denominator = ev_conf + wh_conf
    denominator[denominator == 0] = 1

    ev_conf = np.divide(ev_conf, denominator)
    wh_conf = np.divide(wh_conf, denominator)

    mid_cons_vals[wh_idx][negative_residual] = \
        np.minimum(np.multiply(wh_conf, available_cons), mid_cons_vals[wh_idx])[negative_residual]
    min_cons_vals[wh_idx][negative_residual] = \
        np.minimum(np.multiply(wh_conf, available_cons), min_cons_vals[wh_idx])[negative_residual]
    max_cons_vals[wh_idx][negative_residual] = \
        np.minimum(np.multiply(wh_conf, available_cons), max_cons_vals[wh_idx])[negative_residual]

    mid_cons_vals[ev_idx][negative_residual] = \
        np.minimum(np.multiply(ev_conf, available_cons), mid_cons_vals[ev_idx])[negative_residual]
    min_cons_vals[ev_idx][negative_residual] = \
        np.minimum(np.multiply(ev_conf, available_cons), min_cons_vals[ev_idx])[negative_residual]
    max_cons_vals[ev_idx][negative_residual] = \
        np.minimum(np.multiply(ev_conf, available_cons), max_cons_vals[ev_idx])[negative_residual]

    available_cons[negative_residual] = available_cons[negative_residual] - (mid_cons_vals[ev_idx])[negative_residual]
    available_cons[negative_residual] = available_cons[negative_residual] - (mid_cons_vals[wh_idx])[negative_residual]

    if hvac_adjust:
        mid_cons_vals[get_index(hvac_name)][negative_residual] = \
            np.minimum(available_cons, mid_cons_vals[get_index(hvac_name)])[negative_residual]
        min_cons_vals[get_index(hvac_name)][negative_residual] = \
            np.minimum(available_cons, min_cons_vals[get_index(hvac_name)])[negative_residual]
        max_cons_vals[get_index(hvac_name)][negative_residual] = \
            np.minimum(available_cons, max_cons_vals[get_index(hvac_name)])[negative_residual]

    return mid_cons_vals, min_cons_vals, max_cons_vals, available_cons


def adjust_box_hvac(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals, available_cons, conf_vals, app_name, hvac_name):

    """
    Modify appliance mid/min/max ranges in cases where true disagg appliances are overlapping

    Parameters:
        negative_residual         (np.ndarray)    : tou where disagg residual is negative
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        available_cons            (np.ndarray)    : disagg res
        conf_vals                 (np.ndarray)    : TS level conf vals of all app
        app_name                  (str)           : target app name
        hvac_name                 (str)           : name of hvac appliancee to be adjusted

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
        available_cons            (np.ndarray)    : updated res
    """

    wh_conf = np.fmin(1, conf_vals[get_index(app_name), :, :])

    config = get_inf_config().get("neg_res_handling")

    score_val_bucket_for_box_hvac_adjusment = config.get('score_val_bucket_for_box_hvac_adjusment')
    conf_val_bucket_for_box_hvac_adjusment = config.get('conf_val_bucket_for_box_hvac_adjusment')
    conf_thres_for_non_ev_users = config.get('conf_thres_for_non_ev_users')
    conf_thres_for_ev_users = config.get('conf_thres_for_ev_users')

    allow_hvac = np.array(score_val_bucket_for_box_hvac_adjusment)[np.digitize(wh_conf, conf_val_bucket_for_box_hvac_adjusment, right=True)]

    allow_hvac = 1 - allow_hvac

    available_cons[negative_residual] = available_cons[negative_residual] - np.multiply((mid_cons_vals[get_index(app_name)]), allow_hvac)[negative_residual]
    available_cons[negative_residual] = np.fmax(0, available_cons[negative_residual])

    mid_cons_vals[get_index(hvac_name)][negative_residual] = \
        np.minimum(available_cons, mid_cons_vals[get_index(hvac_name)])[negative_residual]
    min_cons_vals[get_index(hvac_name)][negative_residual] = \
        np.minimum(available_cons, min_cons_vals[get_index(hvac_name)])[negative_residual]
    max_cons_vals[get_index(hvac_name)][negative_residual] = \
        np.minimum(available_cons, max_cons_vals[get_index(hvac_name)])[negative_residual]

    modify_app_bool = np.logical_and(negative_residual, wh_conf < conf_thres_for_non_ev_users)

    if app_name == 'ev':
        modify_app_bool = np.logical_and(negative_residual, np.logical_and(mid_cons_vals[get_index(hvac_name)] > 0, wh_conf < conf_thres_for_ev_users))

    mid_cons_vals[get_index(app_name)][modify_app_bool] = \
        np.minimum(np.multiply((mid_cons_vals[get_index(app_name)]), allow_hvac), mid_cons_vals[get_index(app_name)])[modify_app_bool]
    min_cons_vals[get_index(app_name)][modify_app_bool] = \
        np.minimum(np.multiply((mid_cons_vals[get_index(app_name)]), allow_hvac), min_cons_vals[get_index(app_name)])[modify_app_bool]
    max_cons_vals[get_index(app_name)][modify_app_bool] = \
        np.minimum(np.multiply((mid_cons_vals[get_index(app_name)]), allow_hvac), max_cons_vals[get_index(app_name)])[modify_app_bool]

    return mid_cons_vals, min_cons_vals, max_cons_vals, available_cons


def adjust_box_timed(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals, available_cons,
                     conf_vals, app_name, timed_name):

    """
    Modify appliance mid/min/max ranges in cases where true disagg appliances are overlapping

    Parameters:
        negative_residual         (np.ndarray)    : tou where disagg residual is negative
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        available_cons            (np.ndarray)    : disagg res
        conf_vals                 (np.ndarray)    : TS level conf vals of all app
        app_name                  (str)           : target app name
        timed_name                (str)           : target timed appliance (TWH or PP) used for adjusting negativve residual points

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
        available_cons            (np.ndarray)    : updated res
    """

    pp_conf = np.fmin(1, (1 * (conf_vals[get_index(timed_name)] + 0.3)))

    available_cons[negative_residual] = available_cons[negative_residual] - (pp_conf * mid_cons_vals[get_index(timed_name)])[negative_residual]
    available_cons[negative_residual] = np.fmax(0, available_cons[negative_residual])

    mid_cons_vals[get_index(app_name)][negative_residual] = \
        np.minimum(available_cons, mid_cons_vals[get_index(app_name)])[negative_residual]
    min_cons_vals[get_index(app_name)][negative_residual] = \
        np.minimum(available_cons, min_cons_vals[get_index(app_name)])[negative_residual]
    max_cons_vals[get_index(app_name)][negative_residual] = \
        np.minimum(available_cons, max_cons_vals[get_index(app_name)])[negative_residual]

    return mid_cons_vals, min_cons_vals, max_cons_vals, available_cons


def adjust_hvac_timed(negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals, available_cons, conf_vals, hvac_name, timed_name):

    """
    Modify appliance mid/min/max ranges in cases where true disagg appliances are overlapping

    Parameters:
        negative_residual         (np.ndarray)    : tou where disagg residual is negative
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        available_cons            (np.ndarray)    : disagg res
        conf_vals                 (np.ndarray)    : TS level conf vals of all app
        hvac_name                 (str)           : name of hvac appliancee to be adjusted
        timed_name                (str)           : target timed appliance (TWH or PP) used for adjusting negativve residual points

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
        available_cons            (np.ndarray)    : updated res
    """

    config = get_inf_config().get("neg_res_handling")

    wh_conf =  (conf_vals[get_index(timed_name)])

    score_val_bucket_for_timed_hvac_adjusment = config.get('score_val_bucket_for_timed_hvac_adjusment')
    conf_val_bucket_for_timed_hvac_adjusment = config.get('conf_val_bucket_for_timed_hvac_adjusment')
    conf_thres_for_wh = config.get('conf_thres_for_wh')

    allow_hvac = np.array(score_val_bucket_for_timed_hvac_adjusment)[np.digitize(wh_conf, conf_val_bucket_for_timed_hvac_adjusment, right=True)]

    allow_hvac = 1 - allow_hvac

    available_cons[negative_residual] = available_cons[negative_residual] - np.multiply((mid_cons_vals[get_index(timed_name)]), allow_hvac)[negative_residual]
    available_cons[negative_residual] = np.fmax(0, available_cons[negative_residual])

    mid_cons_vals[get_index(hvac_name)][negative_residual] = \
        np.minimum(available_cons, mid_cons_vals[get_index(hvac_name)])[negative_residual]
    min_cons_vals[get_index(hvac_name)][negative_residual] = \
        np.minimum(available_cons, min_cons_vals[get_index(hvac_name)])[negative_residual]
    max_cons_vals[get_index(hvac_name)][negative_residual] = \
        np.minimum(available_cons, max_cons_vals[get_index(hvac_name)])[negative_residual]

    modify_app_bool = np.logical_and(negative_residual, np.logical_and(mid_cons_vals[get_index(hvac_name)] > 0, wh_conf < conf_thres_for_wh))

    mid_cons_vals[get_index(timed_name)][modify_app_bool] = \
        np.minimum(np.multiply((mid_cons_vals[get_index(timed_name)]), allow_hvac), mid_cons_vals[get_index(timed_name)])[modify_app_bool]
    min_cons_vals[get_index(timed_name)][modify_app_bool] = \
        np.minimum(np.multiply((mid_cons_vals[get_index(timed_name)]), allow_hvac), min_cons_vals[get_index(timed_name)])[modify_app_bool]
    max_cons_vals[get_index(timed_name)][modify_app_bool] = \
        np.minimum(np.multiply((mid_cons_vals[get_index(timed_name)]), allow_hvac), max_cons_vals[get_index(timed_name)])[modify_app_bool]

    return mid_cons_vals, min_cons_vals, max_cons_vals, available_cons


def adjust_hvac_timed_box(month_list, negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals,
                          available_cons, conf_vals, hvac_name, timed_name, box_name):

    """
    Modify appliance mid/min/max ranges in cases where true disagg appliances are overlapping

    Parameters:
        month_list                (np.ndarray)    : month list
        negative_residual         (np.ndarray)    : tou where disagg residual is negative
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        available_cons            (np.ndarray)    : disagg res
        conf_vals                 (np.ndarray)    : TS level conf vals of all app
        hvac_name                 (str)           : name of hvac appliancee to be adjusted
        timed_name                (str)           : target timed appliance (TWH or PP) used for adjusting negativve residual points
        box_name                  (str)           : target box type cons appliance (WH or EV) used for adjusting negativve residual points

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
    """

    # incase of overlap between 3 disagg appliances,
    # monthly adjustment is done to reduce overestimation of ev/wh/hvac/pp based on ts level confidence values

    for month_idx in range(Cgbdisagg.MONTHS_IN_YEAR):

        if not np.sum(month_list == month_idx):
            continue

        target_days = month_list == month_idx

        timed_conf = np.median(conf_vals[get_index(timed_name)][month_list == month_idx][conf_vals[get_index(timed_name)][month_list == month_idx] > 0])
        box_conf = np.median(conf_vals[get_index(box_name)][month_list == month_idx][conf_vals[get_index(box_name)][month_list == month_idx] > 0])
        hvac_conf = np.median(conf_vals[get_index(hvac_name)][month_list == month_idx][conf_vals[get_index(hvac_name)][month_list == month_idx] > 0])

        if  box_name == 'wh':
            wh_idx = get_index(box_name)
            month_cond = month_list == month_idx
            box_conf  = np.median((conf_vals[wh_idx][month_cond][conf_vals[wh_idx][month_cond] >
                                                                 0])[(conf_vals[wh_idx][month_cond][conf_vals[wh_idx][month_cond] > 0]) <= 1])

        if timed_conf == min(hvac_conf, min(box_conf, timed_conf)):
            mid_cons_vals[:, target_days], min_cons_vals[:, target_days], max_cons_vals[:, target_days], available_cons[month_list == month_idx] = \
                adjust_box_hvac(negative_residual[month_list == month_idx], mid_cons_vals[:, target_days], min_cons_vals[:, target_days],
                                max_cons_vals[:, target_days], available_cons[month_list == month_idx],
                                conf_vals[:, target_days], box_name, hvac_name)

            mid_cons_vals[:, target_days], min_cons_vals[:, target_days], max_cons_vals[:, target_days], available_cons[month_list == month_idx] = \
                adjust_hvac_timed(negative_residual[month_list == month_idx], mid_cons_vals[:, target_days],
                                  min_cons_vals[:, target_days], max_cons_vals[:, target_days], available_cons[month_list == month_idx],
                                  conf_vals[:, target_days], hvac_name, timed_name)

        elif hvac_conf == min(hvac_conf, min(box_conf, timed_conf)):

            mid_cons_vals[:, target_days], min_cons_vals[:, target_days], max_cons_vals[:, target_days], available_cons[month_list == month_idx] = \
                adjust_box_timed(negative_residual[target_days],  mid_cons_vals[:, target_days], min_cons_vals[:, target_days],
                                 max_cons_vals[:, target_days], available_cons[month_list == month_idx], conf_vals[:, target_days], box_name, timed_name)

            mid_cons_vals[:, target_days], min_cons_vals[:, target_days], max_cons_vals[:, target_days], available_cons[month_list == month_idx] = \
                adjust_box_hvac(negative_residual[month_list == month_idx], mid_cons_vals[:, target_days], min_cons_vals[:, target_days],
                                max_cons_vals[:, target_days], available_cons[month_list == month_idx], conf_vals[:, target_days], box_name, hvac_name)

        else:

            mid_cons_vals[:, target_days], min_cons_vals[:, target_days], max_cons_vals[:, target_days], available_cons[month_list == month_idx] = \
                adjust_hvac_timed(negative_residual[month_list == month_idx], mid_cons_vals[:, target_days],
                                  min_cons_vals[:, target_days], max_cons_vals[:, target_days], available_cons[month_list == month_idx],
                                  conf_vals[:, target_days], hvac_name, timed_name)

            mid_cons_vals[:, target_days], min_cons_vals[:, target_days], max_cons_vals[:, target_days], available_cons[month_list == month_idx] = \
                adjust_box_hvac(negative_residual[month_list == month_idx], mid_cons_vals[:, target_days], min_cons_vals[:, target_days],
                                max_cons_vals[:, target_days], available_cons[month_list == month_idx],
                                conf_vals[:, target_days], box_name, hvac_name)

    return mid_cons_vals, min_cons_vals, max_cons_vals


def adjust_hvac_ev_wh(month_list, negative_residual, mid_cons_vals, min_cons_vals, max_cons_vals, available_cons, conf_vals, hvac_name):

    """
    Modify appliance mid/min/max ranges in cases where true disagg appliances are overlapping

    Parameters:
        month_list                (np.ndarray)    : month data
        negative_residual         (np.ndarray)    : tou where disagg residual is negative
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        available_cons            (np.ndarray)    : disagg res
        conf_vals                 (np.ndarray)    : TS level conf vals of all app
        hvac_name                 (str)           : name of hvac appliancee to be adjusted

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
    """

    ev_idx = get_index('ev')
    wh_idx = get_index('wh')

    # incase of overlap between 3 disagg appliances,
    # monthly adjustment is done to reduce overestimation of ev/wh/hvac based on ts level confidence values

    for month_idx in range(Cgbdisagg.MONTHS_IN_YEAR):

        if not np.sum(month_list == month_idx):
            continue

        target_days = month_list == month_idx

        ev_conf = np.median(conf_vals[ev_idx][target_days][conf_vals[ev_idx][target_days] > 0])
        wh_conf = np.median((conf_vals[wh_idx][target_days][conf_vals[wh_idx][target_days] > 0])[(conf_vals[wh_idx][target_days][conf_vals[wh_idx][target_days] > 0])<= 1])
        hvac_conf = np.median(conf_vals[get_index(hvac_name)][month_list == month_idx][conf_vals[get_index(hvac_name)][month_list == month_idx] > 0])

        ev_conf = np.nan_to_num(ev_conf)
        wh_conf = np.nan_to_num(wh_conf)
        hvac_conf = np.nan_to_num(hvac_conf)

        if ev_conf == min(hvac_conf, min(wh_conf, ev_conf)):
            mid_cons_vals[:, target_days], min_cons_vals[:, target_days], max_cons_vals[:, target_days], available_cons[month_list == month_idx] = \
                adjust_box_hvac(negative_residual[month_list == month_idx], mid_cons_vals[:, target_days], min_cons_vals[:, target_days],
                                max_cons_vals[:, target_days], available_cons[month_list == month_idx], conf_vals[:, target_days], "wh", hvac_name)

            mid_cons_vals[:, target_days], min_cons_vals[:, target_days], max_cons_vals[:, target_days], available_cons[month_list == month_idx] = \
                adjust_box_hvac(negative_residual[month_list == month_idx], mid_cons_vals[:, target_days], min_cons_vals[:, target_days],
                                max_cons_vals[:, target_days], available_cons[month_list == month_idx], conf_vals[:, target_days], "ev", hvac_name)

        elif hvac_conf == min(hvac_conf, min(wh_conf, ev_conf)):
            mid_cons_vals[:, target_days], min_cons_vals[:, target_days], max_cons_vals[:, target_days], available_cons[month_list == month_idx] = \
                adjust_ev_wh(negative_residual[month_list == month_idx], mid_cons_vals[:, target_days], min_cons_vals[:, target_days],
                             max_cons_vals[:, target_days], available_cons[month_list == month_idx], conf_vals[:, target_days], hvac_name, hvac_adjust=True)

        else:
            mid_cons_vals[:, target_days], min_cons_vals[:, target_days], max_cons_vals[:, target_days],  available_cons[month_list == month_idx] = \
                adjust_box_hvac(negative_residual[month_list == month_idx], mid_cons_vals[:, target_days], min_cons_vals[:, target_days],
                                max_cons_vals[:, target_days], available_cons[month_list == month_idx], conf_vals[:, target_days], "ev", hvac_name)

            mid_cons_vals[:, target_days], min_cons_vals[:, target_days], max_cons_vals[:, target_days], available_cons[month_list == month_idx] = \
                adjust_box_hvac(negative_residual[month_list == month_idx], mid_cons_vals[:, target_days], min_cons_vals[:, target_days],
                                max_cons_vals[:, target_days], available_cons[month_list == month_idx], conf_vals[:, target_days], "wh", hvac_name)

    return mid_cons_vals, min_cons_vals, max_cons_vals


def handle_neg_residual_cases(item_input_object, item_output_object, mid_cons_vals, min_cons_vals,
                              max_cons_vals, conf_vals, input_data):

    """
    Modify appliance mid/min/max ranges in cases where true disagg appliances are overlapping

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid output
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        conf_vals                 (np.ndarray)    : TS level conf vals of all app
        input_data                (np.ndarray)    : user input data

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
    """

    pp_idx = get_index('pp')
    wh_idx = get_index('wh')
    cooling_idx = get_index('cooling')
    heating_idx = get_index('heating')
    ev_idx = get_index('ev')
    ent_idx = get_index('ent')
    ld_idx = get_index('ld')
    cook_idx = get_index('cook')

    negative_residual_without_stat_app = (input_data - mid_cons_vals[get_index('ao')] - mid_cons_vals[pp_idx] -
                                          mid_cons_vals[wh_idx] - mid_cons_vals[ev_idx] -
                                          mid_cons_vals[get_index('ref')] - mid_cons_vals[cooling_idx] -
                                          mid_cons_vals[heating_idx] - mid_cons_vals[get_index('li')]) < 0

    min_cons_vals[ent_idx][negative_residual_without_stat_app] = (min_cons_vals[ent_idx] * 0)[negative_residual_without_stat_app]
    min_cons_vals[ld_idx][negative_residual_without_stat_app] = (min_cons_vals[ld_idx] * 0)[negative_residual_without_stat_app]
    min_cons_vals[cook_idx][negative_residual_without_stat_app] = (min_cons_vals[cook_idx] * 0)[negative_residual_without_stat_app]

    negative_residual_with_stat_app = (input_data - mid_cons_vals[get_index('ao')] - mid_cons_vals[pp_idx] -
                                       mid_cons_vals[wh_idx] - mid_cons_vals[ev_idx] - mid_cons_vals[get_index('ref')] -
                                       mid_cons_vals[cooling_idx] - mid_cons_vals[heating_idx] - mid_cons_vals[get_index('li')] -
                                       mid_cons_vals[get_index('ent')] - mid_cons_vals[get_index('ld')] - mid_cons_vals[get_index('cook')]) < 0

    mid_cons_vals[ent_idx][negative_residual_with_stat_app] = (mid_cons_vals[ent_idx] * 0.3)[negative_residual_with_stat_app]
    mid_cons_vals[ld_idx][negative_residual_with_stat_app] = (mid_cons_vals[ld_idx] * 0.3)[negative_residual_with_stat_app]
    mid_cons_vals[cook_idx][negative_residual_with_stat_app] = (mid_cons_vals[cook_idx] * 0.3)[negative_residual_with_stat_app]

    month = pd.DatetimeIndex(item_output_object.get("date_list")).month.values - 1

    # Handling cases where appliance output overlaps leading to negative residual

    negative_residual = (input_data - np.sum(mid_cons_vals, axis=0)) < 0

    non_signature_cons = mid_cons_vals[get_index('ao')] + mid_cons_vals[get_index('ref')]

    available_cons = np.fmax(0, input_data - non_signature_cons)

    negative_residual = np.logical_and(negative_residual, min_cons_vals[heating_idx] > 0)

    negative_residual_copy = copy.deepcopy(negative_residual)

    if not (int(item_input_object.get("item_input_params").get("timed_wh_user"))) and (not np.all(negative_residual_with_stat_app == 0)):

        mid_cons_vals, min_cons_vals, max_cons_vals = adjust_cons(negative_residual, mid_cons_vals,
                                                                  min_cons_vals, max_cons_vals, available_cons, conf_vals, "heating", month)

        negative_residual = np.logical_and(np.logical_not(negative_residual_copy), input_data - np.sum(mid_cons_vals, axis=0) < 0)

        mid_cons_vals, min_cons_vals, max_cons_vals = adjust_cons(negative_residual, mid_cons_vals,
                                                                  min_cons_vals, max_cons_vals, available_cons, conf_vals, "cooling", month)

    if int(item_input_object.get("item_input_params").get("timed_wh_user")) and (not np.all(negative_residual_with_stat_app == 0)):

        mid_cons_vals, min_cons_vals, max_cons_vals = adjust_cons_for_twh_user(negative_residual,
                                                                               mid_cons_vals, min_cons_vals, max_cons_vals,
                                                                               available_cons, conf_vals, "heating", month)

        negative_residual = np.logical_and(np.logical_not(negative_residual_copy),
                                           input_data - np.sum(mid_cons_vals, axis=0) < 0)

        mid_cons_vals, min_cons_vals, max_cons_vals = adjust_cons_for_twh_user(negative_residual,
                                                                               mid_cons_vals, min_cons_vals, max_cons_vals,
                                                                               available_cons, conf_vals, "cooling", month)

    return mid_cons_vals, min_cons_vals, max_cons_vals
