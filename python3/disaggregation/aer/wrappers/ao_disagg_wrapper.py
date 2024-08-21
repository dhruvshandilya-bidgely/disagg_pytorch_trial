"""
Author - Abhinav Srivastava
Date - 22nd Oct 2018
Call the ao disaggregation wrapper to run ao module
"""

# Import python packages

import copy
import logging
import scipy
import numpy as np
import pandas as pd

from scipy.stats.mstats import mquantiles
from datetime import datetime
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.hvac.init_hourly_hvac_params import init_hvac_params
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params

from python3.utils.write_estimate import write_estimate
from python3.utils.time.get_time_diff import get_time_diff

from python3.disaggregation.aer.ao.compute_baseload import compute_baseload
from python3.disaggregation.aer.ao.compute_baseload_daily import compute_day_level_ao

from python3.utils.validate_output_schema import validate_appliance_profile_schema_for_billcycle


def extract_hsm(disagg_input_object, global_config):

    """Function to extract hsm

    Parameters:
        disagg_input_object (dict)              : Contains input related key attributes
        global_config       (logging object)    : Records logs during code flow

    Returns:
        hsm_in              (dict)              : Dictionary containing ao hsm
        hsm_fail            (bool)              : Boolean indicating whether valid hsm exists or not

    """

    # noinspection PyBroadException
    try:

        hsm_dic = disagg_input_object.get('appliances_hsm')
        hsm_in = hsm_dic.get('ao')

    except KeyError:

        hsm_in = None

    hsm_fail = (hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0) and \
               (global_config.get("disagg_mode") == "mtd")

    return hsm_in, hsm_fail


def extract_epoch_hvac(cooling_estimate, heating_estimate, day_idx, day_ao, epoch_valid_cdd, epoch_valid_hdd, epoch_ao):

    """
    Function to extract epoch level hvac from AO

    Parameters:

        cooling_estimate (np.ndarray)       : Cooling estimates from regression
        heating_estimate (np.ndarray)       : Heating estimates from regression
        day_idx (np.ndarray)                : Day identifier
        day_ao (np.ndarray)                 : Day level ao
        epoch_valid_cdd (np.ndarray)        : Epoch level valid cooling points
        epoch_valid_hdd (np.ndarray)        : Epoch level valid heating points
        epoch_ao (np.ndarray)               : Epoch level AO

    Returns:
        cooling_epoch_output (np.ndarray)   : Epoch level cooling extracted
        heating_epoch_output (np.ndarray)   : Epoch level heating extracted
    """

    df = pd.DataFrame()
    df['cooling_estimate'] = cooling_estimate
    df['heating_estimate'] = heating_estimate
    df['day_ao'] = day_ao

    df['cooling > ao'] = df['cooling_estimate'] > df['day_ao']
    df['heating > ao'] = df['heating_estimate'] > df['day_ao']

    df['cooling_capped'] = df['cooling_estimate']
    df['heating_capped'] = df['heating_estimate']

    df['cooling_capped'][df['cooling > ao']] = df['day_ao']
    df['heating_capped'][df['heating > ao']] = df['day_ao']

    daily_cooling_epoch_count = np.bincount(day_idx, epoch_valid_cdd.astype(int))
    daily_heating_epoch_count = np.bincount(day_idx, epoch_valid_hdd.astype(int))

    cooling_epoch_count = daily_cooling_epoch_count[day_idx]
    heating_epoch_count = daily_heating_epoch_count[day_idx]

    cooling_epoch_output = np.array(df['cooling_capped'])[day_idx] / cooling_epoch_count
    cooling_epoch_output[np.isnan(cooling_epoch_output)] = 0
    heating_epoch_output = np.array(df['heating_capped'])[day_idx] / heating_epoch_count
    heating_epoch_output[np.isnan(heating_epoch_output)] = 0

    cooling_epoch_output[cooling_epoch_output >= epoch_ao] = epoch_ao[cooling_epoch_output >= epoch_ao]
    heating_epoch_output[heating_epoch_output >= epoch_ao] = epoch_ao[heating_epoch_output >= epoch_ao]

    return cooling_epoch_output, heating_epoch_output


def get_valid_degree_days(degree_day, day_ao):

    """
    Function to get valid degree days by removing outliers, just like DS-450

    Parameters:

        degree_day (np.ndarray)         : Array of cdd or hdd
        day_ao (np.ndarray)             : Array of day level AO

    Returns:
        valid_degree_day (np.ndarray)   : Array of valid cdd or hdd points added
    """

    static_params = hvac_static_params()

    valid_degree_day = scipy.ones(np.size(day_ao), float)
    valid_degree_day[degree_day <= 0] = 0
    valid_degree_day[day_ao <= 0] = 0

    degree_day_max = np.mean(degree_day[degree_day > 0]) + 0.5 * np.std(degree_day[degree_day > 0])
    degree_day_min = np.mean(degree_day[degree_day > 0]) - 0.5 * np.std(degree_day[degree_day > 0])

    day_ao_max = np.mean(day_ao[day_ao > 0]) + 0.5 * np.std(day_ao[day_ao > 0])
    day_ao_min = np.mean(day_ao[day_ao > 0]) - 0.5 * np.std(day_ao[day_ao > 0])

    no_degree_day_1 = np.sum((day_ao < day_ao_min) & (degree_day > degree_day_max))
    no_degree_day_2 = np.sum((day_ao > day_ao_max) & (degree_day < degree_day_min))

    max_days_to_remove = static_params['ao']['max_days_to_remove']

    if no_degree_day_1 <= max_days_to_remove:
        valid_degree_day[(day_ao < day_ao_min) & (degree_day > degree_day_max)] = 0

    if no_degree_day_2 <= max_days_to_remove:
        valid_degree_day[(day_ao > day_ao_max) & (degree_day < degree_day_min)] = 0

    return valid_degree_day


def get_ao_cooling_regression(cdd, valid_cdd, day_ao_over_baseload, disagg_output_object, logger_ao):

    """
    Function to perform regression for cooling

    Parameters:

        cdd (np.ndarray)                    : Array of cdd with base of 65 F
        valid_cdd (np.ndarray)              : Array of valid cooling points
        day_ao_over_baseload (np.ndarray)   : Array of ao that is over baseload calculated earlier
        disagg_output_object (dict)         : Dictionary containing all the outputs
        logger_ao (logging object)          : Records progress of algo

    Returns:

        ao_cooling_regression (np.ndarray)  : Array of cooling extracted from regression
    """

    # noinspection PyBroadException
    try:

        logger_ao.info(' Attempting cooling regression. |')
        cooling_regression = LinearRegression().fit(cdd[valid_cdd == 1].reshape(-1, 1), day_ao_over_baseload[valid_cdd == 1].reshape(-1, 1))
        cooling_coefficient = cooling_regression.coef_

        if cooling_coefficient[0][0] <= 0:
            cooling_coefficient = np.array([[0]])

        cool_r_square = pearsonr(cdd[valid_cdd == 1], day_ao_over_baseload[valid_cdd == 1])[0]
        ao_cooling_regression = list(cdd * cooling_coefficient[0])

        logger_ao.info(' Regression part of AO Cooling. Coefficient : {}, R-sq : {} |'.format(cooling_coefficient, cool_r_square))
        disagg_output_object['created_hsm']['ao']['attributes']['ac_coefficient'] = cooling_coefficient[0]

    except Exception:

        logger_ao.info(' By-passing regression. Not enough valid points for ao regression |')
        ao_cooling_regression = list(np.repeat(0, len(cdd)))
        disagg_output_object['created_hsm']['ao']['attributes']['ac_coefficient'] = 0

    return ao_cooling_regression


def get_ao_heating_regression(hdd, valid_hdd, day_ao_over_baseload, disagg_output_object, logger_ao):

    """
    Function to perform regression for heating

    Parameters:

        hdd (np.ndarray)                    : Array of hdd with base of 65 F
        valid_hdd (np.ndarray)              : Array of valid heating points
        day_ao_over_baseload (np.ndarray)   : Array of ao that is over baseload calculated earlier
        disagg_output_object (dict)         : Dictionary containing all the outputs
        logger_ao (logging object)          : Records progress of algo

    Returns:

        ao_heating_regression (np.ndarray)  : Array of heating extracted from regression
    """

    # noinspection PyBroadException
    try:

        logger_ao.info(' Attempting heating regression. |')
        heating_regression = LinearRegression().fit(hdd[valid_hdd == 1].reshape(-1, 1), day_ao_over_baseload[valid_hdd == 1].reshape(-1, 1))
        heating_coefficient = heating_regression.coef_

        if heating_coefficient[0][0] <= 0:
            heating_coefficient = np.array([[0]])

        heat_r_square = pearsonr(hdd[valid_hdd == 1], day_ao_over_baseload[valid_hdd == 1])[0]
        ao_heating_regression = list(hdd * heating_coefficient[0])

        logger_ao.info(' Regression part of AO Heating. Coefficient : {}, R-sq : {} |'.format(heating_coefficient, heat_r_square))
        disagg_output_object['created_hsm']['ao']['attributes']['sh_coefficient'] = heating_coefficient[0]

    except Exception:

        logger_ao.info(' By-passing regression. Not enough valid points for ao regression |')
        ao_heating_regression = list(np.repeat(0, len(hdd)))
        disagg_output_object['created_hsm']['ao']['attributes']['sh_coefficient'] = 0

    return ao_heating_regression


def postprocess_ao_seasonality(disagg_input_object, disagg_output_object, month_epoch, month_idx):

    """
    Function to Take off False positive of AO HVAC

    Parameters:

        disagg_input_object (dict)          : Dictionary containing all the inputs
        disagg_output_object (dict)         : Dictionary containing all the outputs
        month_epoch (np.ndarray)            : Array containing month epochs
        month_idx (np.ndarray)              : Array containing Month indexes

    Returns:
        None
    """

    static_params = hvac_static_params()

    month_identifier = disagg_input_object['input_data'][:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]

    ao_out = disagg_output_object['ao_seasonality']

    epoch_ac = ao_out['epoch_cooling']
    epoch_sh = ao_out['epoch_heating']
    epoch_grey = ao_out['epoch_grey']

    ac_to_suppress = (ao_out['cooling'] < static_params['ao']['suppress_fp_hvac']) & (ao_out['cooling'] > 0)
    sh_to_suppress = (ao_out['heating'] < static_params['ao']['suppress_fp_hvac']) & (ao_out['heating'] > 0)

    if any(ac_to_suppress):

        month_to_suppress = month_epoch[ac_to_suppress]

        for month in month_to_suppress:

            suppress_month_epochs = (month_identifier == month)
            epoch_grey[suppress_month_epochs] = epoch_grey[suppress_month_epochs] + epoch_ac[suppress_month_epochs]
            epoch_ac[suppress_month_epochs] = 0

        disagg_output_object['ao_seasonality']['epoch_cooling'] = epoch_ac

    if any(sh_to_suppress):

        month_to_suppress = month_epoch[sh_to_suppress]

        for month in month_to_suppress:

            suppress_month_epochs = (month_identifier == month)
            epoch_grey[suppress_month_epochs] = epoch_grey[suppress_month_epochs] + epoch_sh[suppress_month_epochs]
            epoch_sh[suppress_month_epochs] = 0

        disagg_output_object['ao_seasonality']['epoch_heating'] = epoch_sh

    disagg_output_object['ao_seasonality']['epoch_grey'] = epoch_grey

    month_ao_cool = np.bincount(month_idx, epoch_ac)
    month_ao_heat = np.bincount(month_idx, epoch_sh)
    month_grey = np.bincount(month_idx, epoch_grey)

    disagg_output_object['ao_seasonality']['cooling'] = month_ao_cool
    disagg_output_object['ao_seasonality']['heating'] = month_ao_heat
    disagg_output_object['ao_seasonality']['grey'] = month_grey


def apply_ao_seasonality(disagg_input_object, disagg_output_object, epoch_baseload, global_config, logger_ao):

    """
    Function to Extract AO - HVAC

    Parameters:

        disagg_input_object (dict)          : Dictionary containing all the inputs
        disagg_output_object (dict)         : Dictionary containing all the outputs
        epoch_baseload (np.ndarray)         : Array containing epoch level baseload consumption
        global_config (dict)                : Dictionary containing all the user level config information
        logger_ao (logging object)          : Records progress of algo

    Returns:
        None
    """

    static_params = hvac_static_params()

    disagg_output_object['ao_seasonality'] = {}

    input_data = disagg_input_object['input_data']

    copy_input_data = copy.deepcopy(input_data)

    epoch_temperature = copy_input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]

    # getting epoch-algo-ao from disagg output object

    ao_out_idx = disagg_output_object.get('output_write_idx_map').get('ao')
    epoch_ao = disagg_output_object['epoch_estimate'][:, ao_out_idx]

    logger_ao.info(' >> Total epochs where (epoch_ao < epoch_baseload) : {} |'.format(np.nansum(epoch_ao <
                                                                                                epoch_baseload)))

    # Handling negative values flowing into seasonality algo

    epoch_baseload[epoch_ao < epoch_baseload] = epoch_ao[epoch_ao < epoch_baseload]
    epoch_ao_over_baseload = epoch_ao - epoch_baseload
    epoch_ao_over_baseload[np.isnan(epoch_ao_over_baseload)] = 0

    # getting unique days and unique day indexes at epoch level
    unique_days, day_idx = np.unique(copy_input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_index=False,
                                     return_inverse=True, return_counts=False)

    epoch_cdd = np.fmax(epoch_temperature - static_params['pivot_F'] - static_params['deg_day_tolerance_ac'], 0)
    epoch_hdd = np.fmax(static_params['pivot_F'] - static_params['deg_day_tolerance_sh'] - epoch_temperature, 0)

    # getting daily consumption aggregates

    cdd = np.around(np.bincount(day_idx, epoch_cdd))
    hdd = np.around(np.bincount(day_idx, epoch_hdd))
    day_ao_over_baseload = np.bincount(day_idx, epoch_ao_over_baseload)

    # ensuring either cdd on hdd exist for a day, as same day cannot have AO Cooling and AO Heating

    cdd[cdd < hdd] = 0
    hdd[hdd < cdd] = 0

    if global_config.get('disagg_mode') == 'mtd':

        ao_hsm = disagg_input_object['appliances_hsm']['ao']['attributes']
        ao_cooling_regression = list(cdd * ao_hsm['ac_coefficient'])
        ao_heating_regression = list(hdd * ao_hsm['sh_coefficient'])

    else:

        valid_cdd = get_valid_degree_days(cdd, day_ao_over_baseload)
        valid_hdd = get_valid_degree_days(hdd, day_ao_over_baseload)

        logger_ao.info('Getting regression part of AO Cooling. Valid cdd : {} |'.format(len(valid_cdd)))
        ao_cooling_regression = \
            get_ao_cooling_regression(cdd, valid_cdd, day_ao_over_baseload, disagg_output_object, logger_ao)

        logger_ao.info('Getting regression part of AO Heating. Valid hdd : {} |'.format(len(valid_hdd)))
        ao_heating_regression = \
            get_ao_heating_regression(hdd, valid_hdd, day_ao_over_baseload, disagg_output_object, logger_ao)

    epoch_valid_cdd = (epoch_cdd > 0) & (epoch_ao_over_baseload > 0)
    epoch_valid_hdd = (epoch_hdd > 0) & (epoch_ao_over_baseload > 0)

    # spreading to epoch

    ao_cooling_reg_epoch, ao_heating_reg_epoch = extract_epoch_hvac(ao_cooling_regression, ao_heating_regression,
                                                                    day_idx, day_ao_over_baseload, epoch_valid_cdd,
                                                                    epoch_valid_hdd, epoch_ao_over_baseload)
    # grey ao : Care 2

    grey_epoch = epoch_ao_over_baseload - (ao_cooling_reg_epoch + ao_heating_reg_epoch)
    grey_epoch[grey_epoch < 0 ] = 0

    logger_ao.info(' Total grey epochs after removal of ao-regression ac/sh from ao-over-baseload : {} |'.format(
        np.sum(grey_epoch > 0)))

    grey_day = np.bincount(day_idx, grey_epoch)

    sampling_rate  = disagg_input_object['config']['sampling_rate']
    ao_degree_day_threshold = static_params['ao_degree_day_threshold'] * Cgbdisagg.SEC_IN_HOUR / sampling_rate
    logger_ao.info(' Threshold degree day for assigning left-over grey AO to hvac : {} |'.format(ao_degree_day_threshold))

    grey_hvac_day = \
        np.c_[unique_days, grey_day, cdd, cdd >= ao_degree_day_threshold, hdd, hdd >= ao_degree_day_threshold]

    # local column identifier in grey hvac day
    cdd_col = 3
    hdd_col = 5

    grey_epoch_cooling_identifier = grey_hvac_day[day_idx, cdd_col].astype(bool)
    grey_epoch_heating_identifier = grey_hvac_day[day_idx, hdd_col].astype(bool)

    ao_cooling_reg_epoch[grey_epoch_cooling_identifier] = \
        ao_cooling_reg_epoch[grey_epoch_cooling_identifier] + grey_epoch[grey_epoch_cooling_identifier]

    grey_epoch[grey_epoch_cooling_identifier] = 0

    ao_heating_reg_epoch[grey_epoch_heating_identifier] = \
        ao_heating_reg_epoch[grey_epoch_heating_identifier] + grey_epoch[grey_epoch_heating_identifier]

    grey_epoch[grey_epoch_heating_identifier] = 0
    logger_ao.info(' Total grey epoch left = Consumption : {}, Epochs : {} |'.format(np.nansum(grey_epoch),
                                                                                     np.sum(grey_epoch > 0)))

    # identifying indexes for handling ao hvac in case of nan temperature
    nan_temp_idx = np.isnan(epoch_temperature)
    ao_cooling_idx = ao_cooling_reg_epoch > 0
    ao_heating_idx = ao_heating_reg_epoch > 0

    # handling ao hvac in case of nan temperature
    grey_epoch[nan_temp_idx & ao_cooling_idx] = ao_cooling_reg_epoch[nan_temp_idx & ao_cooling_idx]
    grey_epoch[nan_temp_idx & ao_heating_idx] = ao_heating_reg_epoch[nan_temp_idx & ao_heating_idx]

    ao_cooling_reg_epoch[nan_temp_idx & ao_cooling_idx] = 0
    ao_heating_reg_epoch[nan_temp_idx & ao_heating_idx] = 0

    # aggregating to month
    month_epoch, _, month_idx = scipy.unique(copy_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_index=True,
                                             return_inverse=True)

    month_ao_cool = np.bincount(month_idx, ao_cooling_reg_epoch)
    month_ao_heat = np.bincount(month_idx, ao_heating_reg_epoch)
    month_baseload = np.bincount(month_idx, epoch_baseload)
    month_grey = np.bincount(month_idx, grey_epoch)

    disagg_output_object['ao_seasonality']['cooling'] = month_ao_cool
    disagg_output_object['ao_seasonality']['heating'] = month_ao_heat
    disagg_output_object['ao_seasonality']['baseload'] = month_baseload
    disagg_output_object['ao_seasonality']['grey'] = month_grey

    disagg_output_object['ao_seasonality']['epoch_cooling'] = ao_cooling_reg_epoch
    disagg_output_object['ao_seasonality']['epoch_heating'] = ao_heating_reg_epoch
    disagg_output_object['ao_seasonality']['epoch_baseload'] = epoch_baseload
    disagg_output_object['ao_seasonality']['epoch_grey'] = grey_epoch

    postprocess_ao_seasonality(disagg_input_object, disagg_output_object, month_epoch, month_idx)


def m2m_stability_on_baseload(input_data, baseload, hvac_params, logger_ao, past_baseload_from_hsm=None, min_baseload_from_hsm=None):

    """
    Function adjusts the algo detected ao, to eliminate possibility of hvac creeping in always on (ao)

    Parameters:

        input_data (np.ndarray)                    : Array of user input data
        baseload (numpy array)                     : Array of epoch level ao estimated in AO module
        hvac_params (dict)                         : Dictionary containing hvac algo related initialized parameters
        past_baseload_from_hsm (int)               : Read from HSM - last ao value seen while stabilizing
        min_baseload_from_hsm (int)                : Read from HSM - minimum epoch level ao seen while stabilizing

    Returns:

        ao (numpy array)                           : Array of epoch level stabilized ao
        last_baseload (int)                        : Last ao value seen while stabilizing
        min_baseload (int)                         : Minimum epoch level ao seen while stabilizing
    """

    logger_ao.info(' Baseload at entry of stability. sum:{}, mean:{}, min:{}, max{} |'.format(np.nansum(baseload),
                                                                                              np.nanmean(baseload),
                                                                                              np.nanmin(baseload),
                                                                                              np.nanmax(baseload)))

    if len(baseload) == 0:
        return baseload, None, None, None

    baseload = copy.deepcopy(baseload)

    # seeing when ao values change at epoch level
    baseload_diff = (np.diff(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX].T) != 0).astype(int)
    rez_data = np.nonzero(baseload_diff)[0]

    # keeping a note of epoch indexes where ao changes
    distinct_baseload_idx = rez_data

    # getting minimum ao at epoch level and setting last ao variable to minimum ao
    if np.sum(baseload > 0) > 0:
        min_baseload = np.array(mquantiles((baseload[baseload > 0]), 0.1, alphap=0.5, betap=0.5))
        last_baseload = min_baseload
    else:
        min_baseload = 0
        last_baseload = 0

    mode_mtd = False

    if min_baseload_from_hsm is not None:

        # adjusting ao in mtd mode after reading from hsm attributes
        logger_ao.info(' getting min and last baseloads in mtd mode from hsm : {} |')
        min_baseload = min_baseload_from_hsm
        last_baseload = past_baseload_from_hsm
        rez_data = np.array(np.nonzero((np.diff(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX].T) != 0).astype(int)))

        # getting indexes wher ao changes
        distinct_baseload_idx = np.array(rez_data[0])
        mode_mtd = True

    logger_ao.info(' MTD mode = {} , minimum baseload : {} , last baseload : {} |'.format(mode_mtd, min_baseload,
                                                                                          last_baseload))
    logger_ao.info(' Total distinct algo-baseload indexes : {} |'.format(len(distinct_baseload_idx)))

    # if ao doesn't change at all at epoch level, no need of adjustment
    if np.all(distinct_baseload_idx == 0):

        logger_ao.info(' Baseload does not change at epoch level, no need of adjustment |')
        return baseload, last_baseload, min_baseload

    threshold = hvac_params['adjustment']['HVAC']['MIN_AMPLITUDE']
    above_threshold_count = 0

    for i in range(distinct_baseload_idx.shape[0] + 1):

        if i == len(distinct_baseload_idx):

            start_idx = distinct_baseload_idx[i - 1] + 1
            end_idx = len(baseload)

        elif i == 0:

            start_idx = 0
            end_idx = distinct_baseload_idx[i] + 1

        else:

            start_idx = distinct_baseload_idx[i - 1] + 1
            end_idx = distinct_baseload_idx[i] + 1

        # measuring how much ao is different from last ao seen
        diff_baseload = baseload[start_idx] - last_baseload

        # if difference in ao jump is above threshold, stabilizing is required
        if diff_baseload > threshold:

            above_threshold_count += 1
            baseload[start_idx:end_idx] = last_baseload

        elif ~mode_mtd and baseload[start_idx] > min_baseload:
            last_baseload = baseload[start_idx]

    logger_ao.info(' Done Stabilizing {} times. Threshold {},  Minimum baseload : {} , Last baseload : {} |'.format(
        above_threshold_count, threshold, min_baseload, last_baseload))

    logger_ao.info(' Baseload at exit of stability. sum:{}, mean:{}, min:{}, max{} |'.format(np.nansum(baseload),
                                                                                             np.nanmean(baseload),
                                                                                             np.nanmin(baseload),
                                                                                             np.nanmax(baseload)))
    return baseload, last_baseload, min_baseload


def ao_disagg_wrapper(disagg_input_object, disagg_output_object):
    """
    Function to estimate always on consumption at epoch level

    Parameters:

        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs

    Returns:

        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # Initiate logger for the ao module
    logger_ao_base = disagg_input_object.get('logger').getChild('ao_disagg_wrapper')
    logger_ao = logging.LoggerAdapter(logger_ao_base, disagg_input_object.get('logging_dict'))
    logger_ao_pass = {"logger": logger_ao_base, "logging_dict": disagg_input_object.get("logging_dict")}

    t_ao_start = datetime.now()

    error_list = []

    global_config = disagg_input_object.get('config')

    if global_config is None:
        error_list.append('Key Error: config does not exist')

    input_data = copy.deepcopy(disagg_input_object.get('input_data'))
    if input_data is None:
        error_list.append('Key Error: input data does not exist')

    exit_status = {
        'exit_code': 1,
        'error_list': error_list,
    }

    # to replicate matlab like results

    is_nan_cons = disagg_input_object.get('data_quality_metrics').get('is_nan_cons')
    input_data[is_nan_cons, Cgbdisagg.INPUT_CONSUMPTION_IDX] = np.nan

    # Extract HSM from disagg input object

    hsm_in, hsm_fail = extract_hsm(disagg_input_object, global_config)

    # Extract sampling rate to send as parameter

    sampling_rate = global_config.get('sampling_rate')

    month_epoch, _, month_idx = scipy.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_index=True,
                                             return_inverse=True)

    disagg_output_object['analytics'] = {'required': True}
    disagg_output_object['analytics']['values'] = {}

    disagg_output_object['ao_seasonality'] = {}

    disagg_output_object['ao_seasonality']['cooling'] = np.zeros(len(month_epoch))
    disagg_output_object['ao_seasonality']['heating'] = np.zeros(len(month_epoch))
    disagg_output_object['ao_seasonality']['baseload'] = np.zeros(len(month_epoch))
    disagg_output_object['ao_seasonality']['grey'] = np.zeros(len(month_epoch))

    disagg_output_object['ao_seasonality']['epoch_cooling'] = np.zeros(len(month_idx))
    disagg_output_object['ao_seasonality']['epoch_heating'] = np.zeros(len(month_idx))
    disagg_output_object['ao_seasonality']['epoch_baseload'] = np.zeros(len(month_idx))
    disagg_output_object['ao_seasonality']['epoch_grey'] = np.zeros(len(month_idx))

    if (global_config.get('run_mode') == 'prod' or global_config.get('run_mode') == 'custom') and (not hsm_fail):

        if global_config.get('disagg_mode') == 'historical':

            disagg_output_object['created_hsm']['ao'] = {}

            logger_ao.info(' ------------------- AO : Month Baseload ------------------------ |')
            month_algo_baseload, epoch_algo_baseload, _ , exit_status = compute_baseload(input_data, sampling_rate,
                                                                                         logger_ao_pass)

            logger_ao.info(' ------------------- AO : Month Stability ------------------------ |')
            epoch_m2m_baseload, last_baseload, min_baseload = \
                m2m_stability_on_baseload(input_data, epoch_algo_baseload[:, 1],init_hvac_params(sampling_rate, disagg_input_object, logger_ao), logger_ao)

            logger_ao.info(' ------------------- AO : DAY LEVEL ------------------------ |')
            month_algo_ao, epoch_algo_ao, epoch_raw_minus_ao, exit_status = compute_day_level_ao(input_data, logger_ao,
                                                                                                 global_config)

            hsm_update = dict({'timestamp': input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX]})
            hsm_update['attributes'] = {}
            hsm_update['attributes']['last_baseload'] = last_baseload
            hsm_update['attributes']['min_baseload'] = min_baseload
            disagg_output_object['created_hsm']['ao'] = hsm_update

        elif global_config.get('disagg_mode') == 'incremental':

            disagg_output_object['created_hsm']['ao'] = {}

            logger_ao.info(' ------------------- AO : Month Baseload ------------------------ |')
            month_algo_baseload, epoch_algo_baseload, _ , exit_status = compute_baseload(input_data, sampling_rate,
                                                                                         logger_ao_pass)

            logger_ao.info(' ------------------- AO : Month Stability ------------------------ |')
            epoch_m2m_baseload, last_baseload, min_baseload = \
                m2m_stability_on_baseload(input_data, epoch_algo_baseload[:, 1], init_hvac_params(sampling_rate, disagg_input_object, logger_ao), logger_ao)

            logger_ao.info(' ------------------- AO : DAY LEVEL ------------------------ |')
            month_algo_ao, epoch_algo_ao, epoch_raw_minus_ao, exit_status = compute_day_level_ao(input_data, logger_ao,
                                                                                                 global_config)

            hsm_update = dict({'timestamp': input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX]})
            hsm_update['attributes'] = {}
            hsm_update['attributes']['last_baseload'] = last_baseload
            hsm_update['attributes']['min_baseload'] = min_baseload
            disagg_output_object['created_hsm']['ao'] = hsm_update

        elif global_config.get('disagg_mode') == 'mtd':

            ao_hsm = disagg_input_object['appliances_hsm']['ao']['attributes']
            last_baseload = ao_hsm['last_baseload']
            min_baseload = ao_hsm['min_baseload']

            logger_ao.info(' ------------------- AO : Month Baseload ------------------------ |')
            month_algo_baseload, epoch_algo_baseload, _ , exit_status = compute_baseload(input_data, sampling_rate,
                                                                                         logger_ao_pass)

            logger_ao.info(' ------------------- AO : Month Stability ------------------------ |')
            epoch_m2m_baseload, last_baseload, min_baseload = \
                m2m_stability_on_baseload(input_data, epoch_algo_baseload[:, 1],init_hvac_params(sampling_rate, disagg_input_object, logger_ao), logger_ao,
                                          last_baseload, min_baseload)

            logger_ao.info(' ------------------- AO : DAY LEVEL ------------------------ |')
            month_algo_ao, epoch_algo_ao, epoch_raw_minus_ao, exit_status = compute_day_level_ao(input_data, logger_ao,
                                                                                                 global_config)

        else:

            logger_ao.error('Unrecognized disagg mode %s |', global_config.get('disagg_mode'))

    if not hsm_fail:

        # Code to write results to disagg output object and Column identifier of ao estimates in epoch_algo_ao

        ao_out_idx = disagg_output_object.get('output_write_idx_map').get('ao')
        read_col_idx = 1

        disagg_output_object = write_estimate(disagg_output_object, epoch_algo_ao, read_col_idx, ao_out_idx, 'epoch')
        disagg_output_object = write_estimate(disagg_output_object, month_algo_ao, read_col_idx, ao_out_idx, 'bill_cycle')

        # Writing the monthly output to log
        monthly_output_log = [(datetime.utcfromtimestamp(month_algo_ao[i, 0]).strftime('%b-%Y'),
                               month_algo_ao[i, read_col_idx]) for i in range(month_algo_ao.shape[0])]

        logger_ao.info("The monthly always on consumption (in Wh) is : | %s", str(monthly_output_log).replace('\n', ' '))

        logger_ao.info(' ------------------- AO : Seasonality ------------------------ |')

        apply_ao_seasonality(disagg_input_object, disagg_output_object, epoch_m2m_baseload, global_config, logger_ao)

    else:

        logger_ao.warning('AO did not run since %s mode required HSM and HSM was missing |',
                          global_config.get('disagg_mode'))

    t_ao_end = datetime.now()
    logger_ao.info('AO Estimation took | %.3f s', get_time_diff(t_ao_start, t_ao_end))

    # Write exit status time taken etc.
    ao_metrics = {
        'time': get_time_diff(t_ao_start, t_ao_end),
        'confidence': 1.0,
        'exit_status': exit_status,
    }

    disagg_output_object['disagg_metrics']['ao'] = ao_metrics

    #Schema Validation for filled appliance profile
    out_bill_cycles = disagg_input_object.get('out_bill_cycles')

    for billcycle_start, _ in out_bill_cycles:
        #TODO(Abhinav): Write your code for filling appliance profile for this bill cycle here
        validate_appliance_profile_schema_for_billcycle(disagg_output_object, billcycle_start,  logger_ao_pass)

    return disagg_input_object, disagg_output_object
