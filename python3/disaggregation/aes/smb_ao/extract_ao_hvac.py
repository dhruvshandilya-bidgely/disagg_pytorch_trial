"""
Author - Abhinav Srivastava
Date - 22nd Oct 2018
Call the ao smb disagg wrapper and get smb ao results
"""

# Import python packages

import copy
import scipy
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aes.smb_ao.ao_hvac_regression import get_ao_heating_regression, get_ao_cooling_regression
from python3.disaggregation.aes.smb_ao.postprocess_ao_hvac import postprocess_ao_seasonality
from python3.disaggregation.aes.smb_ao.ao_hvac_regression import get_valid_degree_days

from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params
from python3.disaggregation.aes.smb_ao.smb_temperature_interpolation import interpolate_temperature
from python3.utils.maths_utils.rolling_function import rolling_function


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


def apply_ao_seasonality_smb(disagg_input_object, disagg_output_object, epoch_baseload, global_config, logger_ao):
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

    ######
    # SMB v2.0 Improvement
    sampling_rate = disagg_input_object.get('config').get('sampling_rate')
    copy_input_data = interpolate_temperature(sampling_rate, copy_input_data, static_params)

    # Modifying the Input Raw Data to be used by all other modules as well.
    input_data[:, Cgbdisagg.INPUT_FEELS_LIKE_IDX] = copy.deepcopy(copy_input_data[:, Cgbdisagg.INPUT_FEELS_LIKE_IDX])
    epoch_temperature = copy_input_data[:, Cgbdisagg.INPUT_FEELS_LIKE_IDX]

    # Identify the season change temperature from season label as opposed to fixing it at 65
    # copy_input_data[:, Cgbdisagg.INPUT_S_LABEL_IDX] = copy_input_data[:, Cgbdisagg.INPUT_S_LABEL_IDX].fillna(limit=3)
    pure_transition_epoch = input_data[:, Cgbdisagg.INPUT_S_LABEL_IDX] == 0
    # If the transition epochs are not present, keep PIVOT_F as fail safe.
    if np.sum(pure_transition_epoch) < 1:
        median_transition_temp = static_params['pivot_F']
        logger_ao.info('Enough season labels not found. Default pivot temperature is: {} F |'.format(
            median_transition_temp))
    else:
        median_transition_temp = np.nanmedian(input_data[:, Cgbdisagg.INPUT_FEELS_LIKE_IDX][pure_transition_epoch])
    logger_ao.info('Season change pivot temperature calculated to be : {} F |'.format(median_transition_temp))
    ################

    epoch_temperature[np.isnan(epoch_temperature)] = median_transition_temp

    # getting epoch-algo-ao from disagg output object
    ao_out_idx = disagg_output_object.get('output_write_idx_map').get('ao_smb')
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

    # SMB v2.0 Improvement
    epoch_cdd = np.fmax(epoch_temperature - median_transition_temp, 0)
    epoch_hdd = np.fmax(median_transition_temp - epoch_temperature, 0)

    # getting daily consumption aggregates
    cdd = np.around(np.bincount(day_idx, epoch_cdd))
    hdd = np.around(np.bincount(day_idx, epoch_hdd))
    day_ao_over_baseload = np.bincount(day_idx, epoch_ao_over_baseload)

    # SMB v2.0 Improvement
    # Get bi-weekly mean of temperatures
    cdd = rolling_function(cdd, Cgbdisagg.DAYS_IN_WEEK, 'mean')
    hdd = rolling_function(hdd, Cgbdisagg.DAYS_IN_WEEK, 'mean')
    # ensuring either cdd on hdd exist for a day, as same day cannot have AO Cooling and AO Heating
    cdd[cdd <= hdd] = 0
    hdd[hdd < cdd] = 0

    # HVAC 3.1 Improvement
    factor = sampling_rate / Cgbdisagg.SEC_IN_HOUR
    cdd[cdd * factor < static_params['day_cdd_low_limit']] = 0
    hdd[hdd * factor < static_params['day_hdd_low_limit']] = 0

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
    grey_epoch[grey_epoch < 0] = 0

    logger_ao.info(' Total grey epochs after removal of ao-regression ac/sh from ao-over-baseload : {} |'.format(
        np.sum(grey_epoch > 0)))

    grey_day = np.bincount(day_idx, grey_epoch)

    ao_degree_day_threshold = static_params['ao_degree_day_threshold'] * Cgbdisagg.SEC_IN_HOUR / sampling_rate
    logger_ao.info(
        ' Threshold degree day for assigning left-over grey AO to hvac : {} |'.format(ao_degree_day_threshold))

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


def m2m_stability_on_baseload(baseload, hvac_params, logger_ao, past_baseload_from_hsm=None,
                              min_baseload_from_hsm=None):
    """
    Function adjusts the algo detected ao, to eliminate possibility of hvac creeping in always on (ao)

    Parameters:

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
    baseload_diff = (np.diff(baseload.T) != 0).astype(int)
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
        rez_data = np.array(np.nonzero((np.diff(baseload) != 0).astype(int)))

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
