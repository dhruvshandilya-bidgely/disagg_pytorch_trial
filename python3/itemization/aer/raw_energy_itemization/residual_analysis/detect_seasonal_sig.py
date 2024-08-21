
"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Calculate hvac consumption from positive residual data
"""

# Import python packages

import copy
import numpy as np
import pandas as pd

# import functions from within the project

import matplotlib.pyplot as plt

from python3.config.pilot_constants import PilotConstants

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import fill_array

from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.raw_energy_itemization.residual_analysis.config.get_residual_config import get_residual_config


def detect_hvac_appliances(item_input_object, item_output_object, sampling_rate, residual, appliance_list, output_data,
                           weather_analytics, input_data, logger):

    """
    detect seasonal signature

    Parameters:
        item_input_object         (dict)        : Dict containing all inputs
        item_output_object        (dict)        : Dict containing all outputs
        sampling_rate             (int)         : sampling rate
        residual                  (np.ndarray)  : residual data
        appliance_list            (list)        : appliance list
        output_data               (np.ndarray)  : disagg output data
        weather_analytics         (dict)        : weather analytics output
        input_data                (np.ndarray)  : input data
        logger                    (logger)      : logger dictionary

    Returns:
        detected_cool             (np.ndarray)  : detected cooling signature
        detected_heat             (np.ndarray)  : detected heating signature
        detected_wh               (np.ndarray)  : detected wh signature
    """

    cooling_index = np.where(np.array(appliance_list) == 'cooling')[0][0] + 1
    heating_index = np.where(np.array(appliance_list) == 'heating')[0][0] + 1
    wh_index = np.where(np.array(appliance_list) == 'wh')[0][0] + 1

    cooling_cons = output_data[cooling_index, :, :]
    heating_cons = output_data[heating_index, :, :]
    wh_cons = np.nan_to_num(copy.deepcopy(output_data[wh_index, :, :]))
    ao_cons = output_data[np.where(np.array(appliance_list) == 'ao')[0][0] + 1, :, :]

    # initializing required inputs for seasonal signature detection like disagg cons, and hvac potential

    if item_input_object.get("item_input_params").get("ao_cool") is not None:
        cooling_cons = cooling_cons - item_input_object.get("item_input_params").get("ao_cool")
        cooling_cons = np.fmax(cooling_cons, 0)
    if item_input_object.get("item_input_params").get("ao_heat") is not None:
        heating_cons = heating_cons - item_input_object.get("item_input_params").get("ao_heat")
        heating_cons = np.fmax(heating_cons, 0)

    if weather_analytics.get('weather') is None or weather_analytics.get('weather').get("season_detection_dict") is None:
        cooling_pot = item_output_object.get("cooling_pot")
        heating_pot = item_output_object.get("heating_pot")
        season_label = copy.deepcopy(item_output_object.get("season"))
    else:
        cooling_pot = weather_analytics.get("weather").get("hvac_potential_dict").get("cooling_pot")
        heating_pot = weather_analytics.get("weather").get("hvac_potential_dict").get("heating_pot")
        season_label = copy.deepcopy(weather_analytics.get('weather').get("season_detection_dict").get("s_label"))

    if np.sum(np.fmax(0, residual)) == 0:
        return np.zeros(cooling_cons.shape), np.zeros(cooling_cons.shape), np.zeros(cooling_cons.shape)

    cooling_cons = np.nan_to_num(cooling_cons)
    heating_cons = np.nan_to_num(heating_cons)

    season_label[np.sum(cooling_pot, axis=1) > 0] = 1
    season_label[np.sum(heating_pot, axis=1) > 0] = -1

    cooling_pot = np.nan_to_num(cooling_pot)
    heating_pot = np.nan_to_num(heating_pot)

    cooling_pot_final = np.zeros((len(cooling_pot), int(Cgbdisagg.HRS_IN_DAY*Cgbdisagg.SEC_IN_HOUR/sampling_rate)))
    heating_pot_final = np.zeros((len(heating_pot), int(Cgbdisagg.HRS_IN_DAY*Cgbdisagg.SEC_IN_HOUR/sampling_rate)))

    samples_per_hour = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    for i in range(len(cooling_pot_final[0])):
        cooling_pot_final[:, i] = cooling_pot[:, int(i/samples_per_hour)]
        heating_pot_final[:, i] = heating_pot[:, int(i/samples_per_hour)]

    detected_cool, detected_heat, detected_wh = \
        seasonal_signature_analysis(item_input_object, item_output_object, input_data, wh_cons, season_label,
                                    [cooling_cons, cooling_pot_final, heating_cons, heating_pot_final],
                                    np.fmax(0, residual), ao_cons, logger)

    return detected_cool, detected_heat, detected_wh


def seasonal_signature_analysis(item_input_object, item_output_object, input_data, wh_cons, season, hvac_params, residual_copy, ao_cons, logger):

    """
    detect seasonal signature in residual data

    Parameters:
        item_input_object         (dict)        : Dict containing all inputs
        item_output_object        (dict)        : Dict containing all outputs
        input_data                (np.ndarray)  : input data
        season                    (np.ndarray)  : seeason tags
        cooling_cons              (np.ndarray)  : disagg cooling cons
        cooling_pot               (np.ndarray)  : cooling potential
        heating_cons              (np.ndarray)  : disagg heating cons
        heating_pot               (np.ndarray)  : heating pot
        residual_copy             (np.ndarray)  : residual data
        ao_cons                   (np.ndarray)  : disagg ao cons

    Returns:
        detected_cool             (np.ndarray)  : detected cooling signature
        detected_heat             (np.ndarray)  : detected heating signature
        detected_wh               (np.ndarray)  : detected wh signature
    """

    cooling_cons = hvac_params[0]
    cooling_pot = hvac_params[1]
    heating_cons = hvac_params[2]
    heating_pot = hvac_params[3]

    # fetch required data

    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    pilot = item_input_object.get("config").get("pilot_id")
    date_list = item_output_object.get("date_list")

    residual = copy.deepcopy(residual_copy)

    samples_per_hour = int(len(cooling_pot[0])/Cgbdisagg.HRS_IN_DAY)

    config = get_residual_config(samples_per_hour).get('hvac_dict')

    japan_pilots = PilotConstants.TIMED_WH_JAPAN_PILOTS

    residual = np.fmin(residual, np.percentile(residual[np.nonzero(residual)], 99))

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    detected_cool = np.zeros(input_data.shape)
    detected_heat = np.zeros(input_data.shape)

    cooling_days = (np.sum(cooling_cons, axis=1) > 0).astype(int)

    non_wh_hours = get_index_array(14*samples_per_hour, 4*samples_per_hour-1, samples_per_hour * Cgbdisagg.HRS_IN_DAY)

    # prepare season tags data

    if np.sum(np.logical_or(season > 0, cooling_days > 0)) < 0.85 * len(season):
        season[cooling_days > 0] = 1

    season_seq = find_seq(season, np.zeros_like(season), np.zeros_like(season), overnight=0)

    for i in range(1, len(season_seq)):
        if season_seq[i, seq_len] < config.get("season_len_thres"):
            season[get_index_array(season_seq[i, seq_start], season_seq[i, seq_end], len(season))] = season_seq[(i-1)%len(season_seq), 0]

    # get yearly energy profile
    base_data = copy.deepcopy(input_data)

    if pilot not in japan_pilots:
        base_data = base_data - ao_cons

    base_data = np.fmax(0, base_data)

    vacation[input_data.sum(axis=1) == 0] = 1

    yearly_profile = get_percentile(base_data, 50, vacation)

    if np.all(yearly_profile == 0):
        return detected_cool, detected_heat, np.zeros_like(detected_heat)

    # prepare season wise profile and season data

    window = 10

    summer_curve, winter_curve, summer_trns, winter_trns, summer, winter, summer_start_trns, winter_start_trns = \
        prepare_season_profiles(season, input_data, vacation, yearly_profile, residual)

    # prepare threshold value for seasonal signature detection

    detection_thres = config.get('hvac_thres')

    heating_thres = max(np.max(winter), np.max(summer)) - min(np.min(winter), np.min(summer))
    heating_thres = min(config.get('max_heat_thres')/samples_per_hour, heating_thres * 0.2)
    heating_thres = max(heating_thres, config.get('min_heat_thres')/samples_per_hour)

    cooling_thres = heating_thres

    swh_pilots = PilotConstants.SEASONAL_WH_ENABLED_PILOTS

    if pilot not in swh_pilots:
        wh_cons[:, :] = 0

    low_hvac_pilots = config.get('low_hvac_pilots')

    if pilot in swh_pilots:
        cooling_thres = cooling_thres * config.get('swh_pilots_multiplier')
        detection_thres = detection_thres * config.get('swh_pilots_multiplier')

    if pilot in low_hvac_pilots:
        cooling_thres = cooling_thres * config.get('low_hvac_multiplier')
        detection_thres = detection_thres * config.get('low_hvac_multiplier')

    # prepare season wise signature from residual data

    # detect cooling

    detected_cool = get_hvac_component(detected_cool, summer_trns, summer_start_trns,  winter_curve,
                                       yearly_profile, cooling_thres, detection_thres, window, pilot)

    detection_thres = config.get('heat_detection_thres')

    # detect heating

    detected_heat = get_hvac_component(detected_heat, winter_trns, winter_start_trns,  summer_curve,
                                       yearly_profile, heating_thres, detection_thres,  window, pilot, wh=-1)

    # detect wh

    if pilot in swh_pilots:
        detection_thres = detection_thres * config.get('swh_multiplier')

    wh_threshold = heating_thres * config.get('wh_multiplier')

    detected_wh = get_hvac_component(np.zeros_like(detected_heat), winter_trns, winter_start_trns, summer_curve,
                                     yearly_profile, wh_threshold, detection_thres,  window, pilot, wh=True)

    detected_wh[:, non_wh_hours] = 0

    low_cons_thres = np.percentile(input_data, 99) / 20
    swh_low_cons_thres = np.percentile(input_data, 99) / 30
    hvac_multiplier = 10

    logger.info('Total detected cooling signature | %s', int(np.sum(detected_cool)))
    logger.info('Total detected heating signature | %s', int(np.sum(detected_heat)))
    logger.info('Total detected WH signature | %s', int(np.sum(detected_wh)))

    # Handle cases where both cooling and heating signatures are getting added on same day during transition months

    detected_cool, detected_heat, overlap_cons, bool_arr = handle_cooling_heating_overlap_days(detected_cool, detected_heat, cooling_cons, heating_cons)

    logger.info('Total detected cooling signature after handling cases of heating/cooling overlap | %s', int(np.sum(detected_cool)))
    logger.info('Total detected heating signature after handling cases of heating/cooling overlap | %s', int(np.sum(detected_heat)))
    logger.info('Total detected WH signature after handling cases of heating/cooling overlap | %s', int(np.sum(detected_wh)))

    # remove high consumption points

    if pilot in swh_pilots:
        detected_cool[detected_cool < swh_low_cons_thres] = 0
    else:
        detected_cool[detected_cool < low_cons_thres] = 0

    detected_heat[detected_heat < low_cons_thres] = 0

    detected_heat = detected_heat * hvac_multiplier
    detected_cool = detected_cool * hvac_multiplier

    detected_heat = np.minimum(detected_heat, residual)
    detected_cool = np.minimum(detected_cool, residual)

    detected_wh[detected_wh < low_cons_thres] = 0

    if np.sum(detected_wh):
        detected_wh[detected_wh < np.percentile(detected_wh[detected_wh > 0], 15)] = 0

    detected_wh = detected_wh * hvac_multiplier
    detected_wh = np.minimum(detected_wh, residual)

    detected_cool[heating_cons.sum(axis=1) > 0] = 0
    detected_heat[cooling_cons.sum(axis=1) > 0] = 0

    # postprocessing of detected signatures

    detected_cool, detected_heat, detected_wh = \
        postprocess_seasonal_output(cooling_cons, heating_cons, pilot, season, bool_arr,
                                    [detected_cool, detected_heat, detected_wh], cooling_pot, heating_pot)


    logger.info('Total detected cooling signature after postprocessing step 1 | %s', int(np.sum(detected_cool)))
    logger.info('Total detected heating signature after postprocessing step 1 | %s', int(np.sum(detected_heat)))
    logger.info('Total detected WH signature after postprocessing step 1 | %s', int(np.sum(detected_wh)))


    detected_cool, detected_heat, detected_wh = \
        postprocess_seasonal_output_to_remove_low_cons(wh_cons, date_list, cooling_cons, heating_cons, pilot, item_input_object,
                                                       detected_cool, detected_heat, detected_wh)

    detected_wh[vacation] = 0

    logger.info('Total detected cooling signature after postprocessing step 2 | %s', int(np.sum(detected_cool)))
    logger.info('Total detected heating signature after postprocessing step 2 | %s', int(np.sum(detected_heat)))
    logger.info('Total detected WH signature after postprocessing step 2 | %s', int(np.sum(detected_wh)))

    return detected_cool, detected_heat, detected_wh


def handle_cooling_heating_overlap_days(detected_cool, detected_heat, cooling_cons, heating_cons):

    """
    Perform 100% itemization

    Parameters:
        detected_cool             (np.ndarray)  : detected cooling signature
        detected_heat             (np.ndarray)  : detected heating signature
        cooling_cons              (np.ndarray)  : disagg cooling cons
        heating_cons              (np.ndarray)  : disagg heating cons

    Returns:
        detected_cool             (np.ndarray)  : detected cooling signature
        detected_heat             (np.ndarray)  : detected heating signature
        detected_wh               (np.ndarray)  : detected wh signature

    """

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    detected_cool_copy = copy.deepcopy(detected_cool)
    detected_heat_copy = copy.deepcopy(detected_heat)

    # checking the cases where both cooling an heating signatures are present

    bool_arr = np.logical_and(detected_cool.sum(axis=1) > 0, detected_heat.sum(axis=1) > 0)

    overlap_cons = np.zeros_like(detected_cool)
    overlap_cons[bool_arr] = copy.deepcopy(detected_cool[bool_arr])

    detected_heat[bool_arr] = 0
    detected_cool[bool_arr] = 0

    detected_heat_copy[np.logical_not(bool_arr)] = 0
    detected_cool_copy[np.logical_not(bool_arr)] = 0

    detected_heat[np.logical_and(detected_heat.sum(axis=1) == 0, heating_cons.sum(axis=1) > 0)] = \
        detected_heat_copy[np.logical_and(detected_heat.sum(axis=1) == 0, heating_cons.sum(axis=1) > 0)]

    detected_cool[np.logical_and(detected_cool.sum(axis=1) == 0, cooling_cons.sum(axis=1) > 0)] = \
        detected_cool_copy[np.logical_and(detected_cool.sum(axis=1) == 0, cooling_cons.sum(axis=1) > 0)]

    overlap_cons[detected_cool.sum(axis=1) > 0] = 0
    overlap_cons[detected_heat.sum(axis=1) > 0] = 0

    # assigning the overlaping signatures to respective category based on disagg cooling/heating days

    cooling_days = (cooling_cons + detected_cool).sum(axis=1) > 0

    cool_seq = find_seq(cooling_days, np.zeros_like(cooling_cons.sum(axis=1)), np.zeros_like(cooling_cons.sum(axis=1)),
                        overnight=0).astype(int)

    for i in range(len(cool_seq)):
        if cool_seq[i, seq_label] > 0 and cool_seq[i, seq_len] < 5:
            cooling_days[cool_seq[i, seq_start]: cool_seq[i, seq_end]] = 0

    heating_days = (heating_cons + detected_heat).sum(axis=1) > 0

    heat_seq = find_seq(heating_days, np.zeros_like(cooling_cons.sum(axis=1)), np.zeros_like(cooling_cons.sum(axis=1)),
                        overnight=0).astype(int)

    for i in range(len(heat_seq)):
        if heat_seq[i, seq_label] > 0 and heat_seq[i, seq_len] < 5:
            heating_days[heat_seq[i, seq_start]: heat_seq[i, seq_end]] = 0

    overlap_seq = find_seq(overlap_cons.sum(axis=1) > 0, np.zeros_like(cooling_cons.sum(axis=1)),
                           np.zeros_like(cooling_cons.sum(axis=1)), overnight=0).astype(int)

    for i in range(len(overlap_seq)):

        if overlap_seq[i, seq_label] > 0:

            thres = 20

            cool_bool = (cooling_days[overlap_seq[i, seq_start] - thres:overlap_seq[i, seq_start]].sum() > thres * 0.5) or \
                        (cooling_days[overlap_seq[i, seq_end]:overlap_seq[i, seq_end] + thres].sum() > thres * 0.5)

            heat_bool = (heating_days[overlap_seq[i, seq_start] - thres:overlap_seq[i, seq_start]].sum() > thres * 0.5) or \
                        (heating_days[overlap_seq[i, seq_end]:overlap_seq[i, seq_end] + thres].sum() > thres * 0.5)

            if cool_bool and not heat_bool:
                detected_cool[overlap_seq[i, seq_start]: overlap_seq[i, seq_end]] = \
                    overlap_cons[overlap_seq[i, seq_start]: overlap_seq[i, seq_end]]

    return detected_cool, detected_heat, overlap_cons, bool_arr


def get_percentile(input_data, window, vacation):

    """
    Calculate rolling avg for the given window size

    Parameters:
        input_data          (np.ndarray)       : input data
        window              (int)              : length of window
        vacation            (np.ndarray)       : vacation days

    Returns:
        data                (np.ndarray)       : output of percentile
    """

    data = np.zeros(input_data.shape)

    perc = 75

    for i in range(0, len(input_data)-window, int(window/2)):

        if np.sum(vacation[i: i + window]) < 4:
            data[i] = np.percentile(input_data[i: i+window], perc, axis=0)

    data = data[~np.all(data == 0, axis=1)]

    if not len(data):
        return np.zeros(len(input_data[0]))

    data = np.percentile(data, 30, axis=0)

    return data


def postprocess_seasonal_output(cooling_cons, heating_cons, pilot, season, bool_arr, seasonal_sig, cooling_pot, heating_pot):

    """
    Perform 100% itemization

    Parameters:
        input_data                (np.ndarray)  : input data
        cooling_cons              (np.ndarray)  : disagg cooling cons
        heating_cons              (np.ndarray)  : disagg heating cons
        pilot                     (int)         : pilot id
        season                    (np.ndarray)  : season tags
        cooling_pot               (np.ndarray)  : cooling potential
        heating_pot               (np.ndarray)  : heating pot

    Returns:
        detected_cool             (np.ndarray)  : detected cooling signature
        detected_heat             (np.ndarray)  : detected heating signature
        detected_wh               (np.ndarray)  : detected wh signature

    """

    detected_cool, detected_heat, detected_wh = seasonal_sig[0], seasonal_sig[1], seasonal_sig[2]

    samples_per_hour = int(detected_cool.shape[1]/Cgbdisagg.HRS_IN_DAY)

    total_samples = int(detected_cool.shape[1])

    config = get_residual_config(samples_per_hour).get('hvac_dict')

    max_hvac_amp = config.get('max_hvac_amp')

    # postprocessing heating and cooling output to handle cases where both consumption are overlapping

    detected_heat[np.tile(np.sum(heating_cons, axis=1) > 0, reps=(total_samples, 1)).T] = \
        np.maximum(detected_heat[np.tile(np.sum(heating_cons, axis=1) > 0, reps=(total_samples, 1)).T],
                   copy.deepcopy(detected_cool[np.tile(np.sum(heating_cons, axis=1) > 0, reps=(total_samples, 1)).T]))

    detected_cool[np.tile(np.sum(cooling_cons, axis=1) > 0, reps=(total_samples, 1)).T] =\
        np.maximum(detected_cool[np.tile(np.sum(cooling_cons, axis=1) > 0, reps=(total_samples, 1)).T],
                   copy.deepcopy(detected_heat[np.tile(np.sum(cooling_cons, axis=1) > 0, reps=(total_samples, 1)).T]))

    detected_heat[np.tile(np.sum(cooling_cons, axis=1) > 0, reps=(total_samples, 1)).T] = 0

    detected_heat_frac = np.zeros_like(detected_heat)

    detected_heat[:,  config.get('zero_heat_hours')] = 0

    if not np.all(detected_heat == 0):
        detected_heat_frac = detected_heat / np.percentile(detected_heat[detected_heat > 0], 98)

    # update cooling and heating output based on distribution of hvac potential

    cooling_pot_copy = copy.deepcopy(cooling_pot)
    heating_pot_copy = copy.deepcopy(heating_pot)

    samples_per_hour = int(samples_per_hour)

    for i in range(samples_per_hour * Cgbdisagg.HRS_IN_DAY):
        cooling_pot[:, i] = np.maximum(cooling_pot[:, i],
                                       np.sum(cooling_pot_copy[:, get_index_array(i-7*samples_per_hour, i+7*samples_per_hour, samples_per_hour * Cgbdisagg.HRS_IN_DAY)], axis=1))

    cooling_pot[heating_pot > 0] = 0
    cooling_pot[np.logical_and(season[:, None] > 0, heating_pot == 0)] = 0.7

    detected_cool[heating_cons > 0] = 0
    detected_heat[cooling_cons > 0] = 0

    # decrease hvac output where hvac potential is lower

    for i in range(samples_per_hour * Cgbdisagg.HRS_IN_DAY):
        neighbouring_window = get_index_array(i-1 * samples_per_hour, i+1 * samples_per_hour, samples_per_hour * Cgbdisagg.HRS_IN_DAY)
        heating_pot[:, i] = np.maximum(heating_pot[:, i], np.sum(heating_pot_copy[:, neighbouring_window], axis=1))

    heating_pot = np.fmin(1, heating_pot)
    detected_heat[detected_heat_frac > heating_pot] = np.multiply(heating_pot[detected_heat_frac > heating_pot], detected_heat[detected_heat_frac > heating_pot])

    # remove hvac consumption lower than a certain threshold

    detected_heat[np.logical_and(heating_cons == 0, detected_heat < config.get('hvac_min_cons')/samples_per_hour)] = 0

    swh_pilots =  PilotConstants.SEASONAL_WH_ENABLED_PILOTS

    thres = config.get('swh_pilot_hvac_min_cons') * (pilot in swh_pilots) + config.get('hvac_min_cons') * (pilot not in swh_pilots)

    detected_cool[np.logical_and(cooling_cons == 0, detected_cool < thres / samples_per_hour)] = 0

    detected_heat[np.logical_and(heating_cons == 0, detected_heat > config.get('hvac_max_cons')/samples_per_hour)] = 0
    detected_cool[np.logical_and(cooling_cons == 0, detected_cool > config.get('hvac_max_cons')/samples_per_hour)] = 0

    detected_cool[detected_cool > max(max_hvac_amp/samples_per_hour, np.percentile(cooling_cons, 97) * 2)] = 0
    detected_heat[detected_heat > max(max_hvac_amp/samples_per_hour, np.percentile(heating_cons, 97) * 2)] = 0

    return  detected_cool, detected_heat, detected_wh


def apply_bill_cycle_limit(detected_cool, detected_heat, detected_wh, temp_heating, temp_cooling, samples_per_hour,
                           pilot, item_input_object):
    """
    Perform 100% itemization

    Parameters:
        detected_cool             (np.ndarray)  : detected cooling signature
        detected_heat             (np.ndarray)  : detected heating signature
        detected_wh               (np.ndarray)  : detected wh signature
        temp_cooling              (np.ndarray)  : disagg cooling
        temp_heating              (np.ndarray)  : disagg heating
        samples_per_hour          (int)         : samples per hour
        pilot                     (int)         : pilot id
        item_input_object         (dict)        : Dict containing all inputs

    Returns:
        detected_cool             (np.ndarray)  : detected cooling signature
        detected_heat             (np.ndarray)  : detected heating signature
        detected_wh               (np.ndarray)  : detected wh signature
    """

    # apply bc level threshold on hvac signature

    swh_pilots = PilotConstants.INDIAN_PILOTS

    total_samples = Cgbdisagg.HRS_IN_DAY * samples_per_hour

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    if pilot in swh_pilots:
        valid_days = np.sum(
            detected_cool[:, get_index_array(20 * samples_per_hour, 4 * samples_per_hour, total_samples)], axis=1) > 0
        detected_cool[valid_days == 0] = 0

    wh_bc_cons = np.zeros(len(unique_bc))

    for i in range(len(unique_bc)):
        if np.sum(detected_heat[bc_list == unique_bc[i]][temp_heating[bc_list == unique_bc[i]] == 0] > 0) < 30*samples_per_hour:
            detected_heat[np.logical_and(temp_heating == 0, (bc_list == unique_bc[i])[:, None])] = 0
        if np.sum(detected_cool[bc_list == unique_bc[i]][temp_cooling[bc_list == unique_bc[i]] == 0] > 0) < 30*samples_per_hour:
            detected_cool[np.logical_and(temp_cooling == 0, (bc_list == unique_bc[i])[:, None])] = 0

        wh_bc_cons[i] = np.sum(detected_wh[bc_list == unique_bc[i]]) * 30 / np.sum(bc_list == unique_bc[i])

    val3 = 5000 * (pilot in swh_pilots) + 10000 * (pilot not in swh_pilots)

    if np.max(wh_bc_cons) < val3:
        detected_wh = np.zeros_like(detected_heat)

    return detected_cool, detected_heat, detected_wh


def postprocess_seasonal_output_to_remove_low_cons(wh_cons, date_list, cooling_cons, heating_cons, pilot, item_input_object, detected_cool, detected_heat, detected_wh):

    """
    Perform 100% itemization

    Parameters:
        date_list                 (np.ndarray)  : date list
        cooling_cons              (np.ndarray)  : disagg cooling cons
        heating_cons              (np.ndarray)  : disagg heating cons
        pilot                     (int)         : pilot id
        item_input_object         (dict)        : Dict containing all inputs
        cooling_pot               (np.ndarray)  : cooling potential
        heating_pot               (np.ndarray)  : heating pot

    Returns:
        detected_cool             (np.ndarray)  : detected cooling signature
        detected_heat             (np.ndarray)  : detected heating signature
        detected_wh               (np.ndarray)  : detected wh signature
    """

    samples_per_hour = int(detected_cool.shape[1]/Cgbdisagg.HRS_IN_DAY)

    config = get_residual_config(samples_per_hour).get('hvac_dict')

    swh_pilots =  PilotConstants.SEASONAL_WH_ENABLED_PILOTS

    temp_data = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_TEMPERATURE_IDX]

    # update hvac consumption based on temperaturee data

    if pilot not in swh_pilots:
        detected_cool[temp_data <= config.get('temp_thres')] = 0
        detected_wh[:, 14*samples_per_hour:] = 0

    detected_heat[temp_data >= config.get('heat_temp_thres')] = 0

    temp_heating = heating_cons
    temp_cooling = cooling_cons

    if item_input_object.get("item_input_params").get("ao_heat") is not None:
        temp_heating = heating_cons - item_input_object.get("item_input_params").get("ao_heat")

    if item_input_object.get("item_input_params").get("ao_cool") is not None:
        temp_cooling = cooling_cons - item_input_object.get("item_input_params").get("ao_cool")

    # apply bc level threshold

    detected_cool, detected_heat, detected_wh = \
        apply_bill_cycle_limit(detected_cool, detected_heat, detected_wh, temp_heating, temp_cooling,
                               samples_per_hour, pilot, item_input_object)

    if np.sum(detected_heat) < 10000:
        detected_heat = np.zeros_like(detected_heat)

    thres = config.get('swh_min_thres') * (pilot in swh_pilots) + config.get('wh_min_thres') * (pilot not in swh_pilots)

    # update wh output for low consumption point

    detected_wh[detected_wh < thres/samples_per_hour] = 0

    detected_cool[detected_wh > 0] = 0

    month_list = pd.DatetimeIndex(date_list).month.values

    # remove wh output if not detected winter season

    if (np.any(np.isin(month_list, config.get('winter_months'))) and
        (np.sum((detected_wh+wh_cons)[np.isin(month_list, config.get('winter_months'))]) == 0)) or \
            (detected_wh.sum() < config.get('swh_monthly_thres')):
        detected_wh[:] = 0

    if np.sum(heating_cons) == 0:
        detected_heat = np.zeros_like(detected_heat)
    if np.sum(cooling_cons) == 0:
        detected_cool = np.zeros_like(detected_cool)

    return detected_cool, detected_heat, detected_wh


def get_hvac_component(detected_sig, summer_trns, summer_start_trns, winter_curve, yearly_profile,
                       hvac_det_thres, limit, window, pilot, wh=0):

    """
    determine hvac/wh component residual data

    Parameters:
        detected_sig                (np.ndarray)        : seasonal signature
        summer_trns                 (np.ndarray)        : summer
        summer_start_trns           (np.ndarray)        : indexes of summer chunks
        winter_curve                (np.ndarray)        : winter profile
        yearly_profile              (np.ndarray)        : yearly energy profile
        pilot                       (int)               : pilot id

    Returns:
        detected_sig                (np.ndarray)        : seasonal signature
    """

    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END

    temp_amp = np.zeros((window, len(detected_sig[0])))

    # determine possible chunks for hvac detection

    active_summer = np.logical_and(np.abs(summer_trns - winter_curve[None, :]) > hvac_det_thres, np.abs(summer_trns - yearly_profile[None, :]) > hvac_det_thres) * (wh == -1) + \
                    np.logical_or(np.abs(summer_trns - winter_curve[None, :]) > hvac_det_thres, np.abs(summer_trns - yearly_profile[None, :]) > hvac_det_thres) * (wh != -1)

    active_summer = active_summer.astype(bool)

    amplitude_arr = summer_trns - yearly_profile[None, :]
    amplitude_arr = np.fmax(0, amplitude_arr)

    samples_per_hour = int(len(detected_sig[0]) / Cgbdisagg.HRS_IN_DAY)

    config = get_residual_config(samples_per_hour).get('hvac_dict')

    swh_pilots = PilotConstants.SEASONAL_WH_ENABLED_PILOTS

    total_samples = Cgbdisagg.HRS_IN_DAY * samples_per_hour

    bool_arr = np.percentile(amplitude_arr, 60, axis=1) < limit/samples_per_hour

    amplitude_arr[np.logical_and(amplitude_arr < np.percentile(amplitude_arr, 50, axis=1)[:, None], bool_arr[:, None])] = 0

    active_summer = np.logical_and(active_summer, amplitude_arr > 0)

    windows = len(summer_trns)

    windows = int(windows * (np.sum(winter_curve) > 0))

    # for each chunk determine the hvac/wh tou

    for i in range(windows):

        temp_amp[:, :] = amplitude_arr[i]

        seq = find_seq(active_summer[i], np.zeros(total_samples), np.zeros(total_samples))

        seq_label = seq[:, 0]
        seq_len = seq[:, 3]

        # hvac detection
        if wh != 1:

            if np.sum(active_summer[i]) < config.get('hvac_min_len') * samples_per_hour:
                continue

            val = (0.75 * samples_per_hour) * (wh == 0) + (1 * samples_per_hour) * (wh != 0)

            seq[np.logical_and(seq_label == 0,  seq_len <= val), 0] = 1

            for j in range(len(seq)):
                active_summer[i] = fill_array(active_summer[i], seq[j, seq_start], seq[j, seq_end], seq[j, 0])

            seq = find_seq(active_summer[i], np.zeros(Cgbdisagg.HRS_IN_DAY * samples_per_hour), np.zeros(Cgbdisagg.HRS_IN_DAY * samples_per_hour))

            seq_label = seq[:, 0]
            seq_len = seq[:, 3]

            val = (config.get('cool_len_thres') * samples_per_hour) * (wh == -1) + (config.get('heat_len_thres') * samples_per_hour) * (wh != -1)

            seq[np.logical_and(seq_label == 1,  seq_len <= val), 0] = 0

        # wh detection
        else:

            if ((np.sum(active_summer[i]) < 0.5 * samples_per_hour) or (np.sum(active_summer[i]) > 18 * samples_per_hour)) or \
                    ((np.sum(active_summer[i, 5*samples_per_hour:]) > 10 * samples_per_hour) and (pilot in swh_pilots)):
                continue

            seq[np.logical_and(seq_label == 1,  seq_len < 0.5 * samples_per_hour), 0] = 0

            val = config.get('wh_len_thres') * (pilot in swh_pilots) + config.get('swh_len_thres') * (pilot not in swh_pilots)

            seq[np.logical_and(seq_label == 1,  seq_len > val * samples_per_hour), 0] = 0

        for j in range(len(seq)):
            active_summer[i] = fill_array(active_summer[i], seq[j, seq_start], seq[j, seq_end], seq[j, 0])

        detected_sig[summer_start_trns[i]: summer_start_trns[i] + window, active_summer[i].astype(bool)] =\
            np.maximum(temp_amp[:, active_summer[i].astype(bool)], detected_sig[summer_start_trns[i]: summer_start_trns[i] + window, active_summer[i].astype(bool)])

    return detected_sig


def prepare_season_profiles(season, input_data, vacation, data, residual):

    """
    prepare season profiles

    Parameters:
        season                      (np.ndarray)        : season info
        input_data                  (np.ndarray)        : input data
        vacation                    (np.ndarray)        : vacation data
        residual                    (np.ndarray)        : residual data

    Returns:
        summer_curve                (np.ndarray)        : summer profile
        winter_curve                (np.ndarray)        : winter profile
        summer_trns                 (np.ndarray)        : summer chunks
        winter_trns                 (np.ndarray)        : winter chunks
        summer                      (np.ndarray)        : individual profile summer chunks
        winter                      (np.ndarray)        : individual profile winter chunks
        summer_start_trns           (np.ndarray)        : indexes of summer chuncks
        winter_start_trns           (np.ndarray)        : indexes of winter chucks
    """

    # determine winter and summer chunks

    summer = np.zeros(input_data.shape)
    winter = np.zeros(input_data.shape)

    summer_trns = np.zeros(input_data.shape)
    winter_trns = np.zeros(input_data.shape)
    summer_start_trns = []
    winter_start_trns = []

    summer_start = []
    winter_start = []

    window = 10

    # prepare season tags individually for summer and winter profile

    season_1 = copy.deepcopy(season)
    season_2 = copy.deepcopy(season)

    season_1[season_1 > 0] = 1
    season_1[season_1 < 0] = -1

    season_2[season_2 > 0] = 1
    season_2[season_2 < 0] = -1

    season_1_seq = find_seq(season_1, np.zeros_like(season_1), np.zeros_like(season_1))
    season_2_seq = find_seq(season_1, np.zeros_like(season_1), np.zeros_like(season_1))

    for i in range(len(season_1_seq)):
        if season_1_seq[i, 0] == 1:
            season_1[get_index_array(season_1_seq[i, 1] - 30, season_1_seq[i, 1], len(season_1))] = 1
            season_1[get_index_array(season_1_seq[i, 2], season_1_seq[i, 2] + 30, len(season_1))] = 1

    for i in range(len(season_2_seq)):
        if season_2_seq[i, 0] == -1:
            season_2[get_index_array(season_2_seq[i, 1] - 30, season_2_seq[i, 1], len(season_1))] = -1
            season_2[get_index_array(season_2_seq[i, 2], season_2_seq[i, 2] + 30, len(season_1))] = -1

    c1 = 0
    c2 = 0

    # fetch season info of individual chunks

    for i in range(0, len(input_data)-window, 10):

        if np.sum(season_1[i: i + window]) > 0 and np.sum(vacation[i: i + window]) < 8:
            summer[c1] = np.percentile(residual[i: i+window], 80, axis=0)
            summer_start.append(i)
            c1 = c1 + 1
        if np.sum(season_2[i: i + window]) < 0 and np.sum(vacation[i: i + window]) < 8:
            winter[c2] = np.percentile(residual[i: i+window], 80, axis=0)
            winter_start.append(i)
            c2 = c2 + 1

    summer = summer[:c1]
    winter = winter[:c2]

    summer_curve, winter_curve, summer, winter, summer_trns, winter_trns, summer_start_trns, winter_start_trns = \
        postprocess_season_profiles(summer, winter, input_data, data, [summer_trns, winter_trns, summer_start_trns, winter_start_trns],
                                    vacation, residual, season_1, season_2)

    return summer_curve, winter_curve, summer_trns, winter_trns, summer, winter, summer_start_trns, winter_start_trns


def postprocess_season_profiles(summer, winter, input_data, yearly_profile, params, vacation, residual, season_1, season_2):

    """
    prepare season profiles

    Parameters:
        summer                      (np.ndarray)        : individual profile summer chunks
        winter                      (np.ndarray)        : individual profile winter chunks
        input_data                  (np.ndarray)        : input data
        yearly_profile              (np.ndarray)        : yearly energy profile
        vacation                    (np.ndarray)        : vacation data
        residual                    (np.ndarray)        : residual data
        season_1                    (np.ndarray)        : season tags for summer profile
        season_2                    (np.ndarray)        : season tags for winter profile

    Returns:
        summer_curve                (np.ndarray)        : summer profile
        winter_curve                (np.ndarray)        : winter profile
        summer_trns                 (np.ndarray)        : summer chunks
        winter_trns                 (np.ndarray)        : winter chunks
        summer                      (np.ndarray)        : individual profile summer chunks
        winter                      (np.ndarray)        : individual profile winter chunks
        summer_start_trns           (np.ndarray)        : indexes of summer chuncks
        winter_start_trns           (np.ndarray)        : indexes of winter chucks
    """

    summer_trns, winter_trns, summer_start_trns, winter_start_trns = params[0], params[1], params[2], params[3]

    c1 = 0
    c2 = 0

    window = 10

    for i in range(0, len(input_data)-window, 10):

        if np.sum(season_1[i: i + window]) >= 0 and np.sum(vacation[i: i + window]) < 8:
            summer_trns[c1] = np.percentile(residual[i: i+window], 80, axis=0)
            summer_start_trns.append(i)
            c1 = c1 + 1
        if np.sum(season_2[i: i + window]) <= 0 and np.sum(vacation[i: i + window]) < 8:
            winter_trns[c2] = np.percentile(residual[i: i+window], 80, axis=0)
            winter_start_trns.append(i)
            c2 = c2 + 1

    summer_trns = summer_trns[:c1]
    winter_trns = winter_trns[:c2]

    if len(summer) == 0:
        summer = np.zeros(len(input_data[0]))

    if len(winter) == 0:
        winter = np.zeros(len(input_data[0]))

    if len(summer):
        summer_curve = np.percentile(summer, Cgbdisagg.DAYS_IN_MONTH, axis=0)
    else:
        summer_curve = yearly_profile

    if len(winter):
        winter_curve = np.percentile(winter, Cgbdisagg.DAYS_IN_MONTH, axis=0)
    else:
        winter_curve = yearly_profile

    if isinstance(winter_curve, float) or isinstance(winter_curve, int):
        winter_curve = np.zeros(len(input_data[0]))

    if isinstance(summer_curve, float) or isinstance(summer_curve, int):
        summer_curve = np.zeros(len(input_data[0]))

    return summer_curve, winter_curve, summer, winter, summer_trns, winter_trns, summer_start_trns, winter_start_trns
