
"""
Author - Nisha Agarwal
Date - 10th Mar 2021
Calculate hvac ts level confidence and potential values
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import fill_array
from python3.itemization.aer.functions.itemization_utils import resample_day_data

from python3.itemization.aer.raw_energy_itemization.utils import get_daily_profile

from python3.itemization.aer.raw_energy_itemization.appliance_potential.config.get_pot_conf import get_pot_conf


def get_hvac_potential(cooling_index, heating_index, item_input_object, item_output_object,
                       sampling_rate, vacation, cool_ao, heat_ao, logger_pass):

    """
    Calculate cooking confidence and potential values

    Parameters:
        cooling_index               (int)           : Index of cooling in the appliance list
        heating_index               (int)           : Index of cooling in the appliance list
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        sampling_rate               (int)           : sampling rate
        vacation                    (np.ndarray)    : array of vacation days
        cool_ao                     (np.ndarray)    : cooling AO component
        heat_ao                     (np.ndarray)    : heating AO component
        logger_pass                 (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)          : updated Dict containing all hybrid outputs
    """

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    t_start = datetime.now()

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_hvac_potential')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Fetching required inputs

    cooling_disagg = item_output_object.get("updated_output_data")[cooling_index, :, :]
    heating_disagg = item_output_object.get("updated_output_data")[heating_index, :, :]
    dow = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_DOW_IDX, :, 0]
    original_input_data = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    # Fetch disagg HVAC AO values

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')

    config = get_pot_conf(samples_per_hour).get('hvac')

    perenial_score_thres = config.get('perenial_score_thres')
    conf_score_weights_for_perenial_usage_user = config.get('conf_score_weights_for_perenial_usage_user')
    conf_score_weights = config.get('conf_score_weights')
    pot_score_weights = config.get('pot_score_weights')
    thres_for_low_hvac_points_adjusment = config.get('thres_for_low_hvac_points_adjusment')
    thres_for_perennial_cons_score_adjusment = config.get('thres_for_perennial_cons_score_adjusment')
    cooling_score_offset = config.get('cooling_score_offset')

    cool_cap = 1

    if not np.all(cooling_disagg == 0):
        cool_cap = np.percentile(cooling_disagg[cooling_disagg > 0], 98)

    heat_cap = 1

    if not np.all(heating_disagg == 0):
        heat_cap = np.percentile(heating_disagg[heating_disagg > 0], 98)

    # removing ao component from HVAC disagg before calculating hvac ts level potential
    # this will be added in the later steps

    if cool_ao is not None:
        cooling_disagg = cooling_disagg - cool_ao

    else:
        cool_ao = np.zeros(cooling_disagg.shape)

    if heat_ao is not None:
        heating_disagg = heating_disagg - heat_ao

    else:
        heat_ao = np.zeros(heating_disagg.shape)

    if np.all((cooling_disagg + heating_disagg) == 0):
        logger.info('No on demand component left in HVAC disagg | ')
        return item_output_object

    weekend_usage_score = np.ones(cooling_disagg.shape)

    season = item_output_object.get("season")

    # HVAC usage potential - potential derived from weather analytics module

    hvac_usage_pot = item_output_object.get("cooling_pot") + item_output_object.get("heating_pot")

    hvac_usage_pot = np.nan_to_num(hvac_usage_pot)

    hvac_output = cooling_disagg + heating_disagg

    # upsampling of hvac potential data

    hvac_usage_pot = resample_day_data(hvac_usage_pot, samples_per_hour * Cgbdisagg.HRS_IN_DAY) / np.max(hvac_usage_pot)

    window = 10

    daily_profile = get_daily_profile(original_input_data, window)

    summer_profile, transition_profile, winter_profile = get_seasonal_profile(original_input_data, season, window)

    # preparing weekday/weekend profile
    # and preparing hvac usage score based on extra weekend usage

    weekend_days = np.logical_or(dow == 1, dow == 7)
    weekday_profile, weekend_profile = get_weekday_profile(hvac_output, dow)

    extra_weekend_tou = np.divide(weekend_profile - weekday_profile, weekday_profile) > 0.6

    seq = find_seq(extra_weekend_tou, np.zeros(len(extra_weekend_tou)), np.zeros(len(extra_weekend_tou)))

    for i in range(len(seq)):
        if seq[i, seq_label] == 1 and seq[i, seq_len] <= 2 * samples_per_hour:
            extra_weekend_tou = fill_array(extra_weekend_tou, seq[i, seq_start], seq[i, seq_end], 0)

    weekend_usage_score_copy = np.ones(cooling_disagg.shape)

    weekend_usage_score_copy[:, extra_weekend_tou] = 1 - np.divide(weekend_profile[extra_weekend_tou] -
                                                                   weekday_profile[extra_weekend_tou],
                                                                   weekday_profile[extra_weekend_tou])

    weekend_usage_score[weekend_days] = weekend_usage_score_copy[weekend_days]

    hvac_output_with_ao = copy.deepcopy(hvac_output + heat_ao + cool_ao)
    hvac_output_with_ao[hvac_output == 0] = 0

    # and preparing hvac usage score based on strike type usage

    strikes_score = 1 - np.divide(np.fmax(0, hvac_output - np.roll(hvac_output, 1, axis=0)), hvac_output_with_ao) * 0.5 - \
                     np.divide(np.fmax(0, hvac_output - np.roll(hvac_output, -1, axis=0)), hvac_output_with_ao) * 0.5

    # and preparing hvac usage score based on perennial usage

    perennial_usage_array = np.zeros(cooling_disagg.shape)
    perennial_usage_array[:, None] = daily_profile

    perennial_usage_score = np.fmin(1, np.divide(np.fmax(0, hvac_output_with_ao - perennial_usage_array), hvac_output_with_ao))

    perennial_usage_score[hvac_output_with_ao == 0] = 0

    # and preparing hvac usage score based on seasonal usage

    transition_day_data = np.zeros(cooling_disagg.shape)

    transition_day_data[:, None] = transition_profile

    summer_extra = np.fmax(0, original_input_data - transition_day_data)
    winter_extra = original_input_data - transition_day_data

    summer_usage_score = np.fmin(1, 0.2 + np.divide(summer_extra, original_input_data))
    winter_usage_score = np.fmin(1, 0.2 + np.divide(winter_extra, original_input_data))
    summer_usage_score[np.sum(cooling_disagg, axis=1) == 0] = 0
    winter_usage_score[np.sum(heating_disagg, axis=1) == 0] = 0
    seasonal_usage_score = winter_usage_score + summer_usage_score

    hvac_usage_pot, weekend_usage_score, perennial_usage_score, seasonal_usage_score, strikes_score =\
        postprocess_hvac_components(hvac_output, hvac_usage_pot, weekend_usage_score, perennial_usage_score,
                                    seasonal_usage_score, strikes_score)

    weekend_usage_score = np.fmax(0, weekend_usage_score)

    if np.all(hvac_usage_pot == 0):
        hvac_usage_pot = np.ones(hvac_usage_pot.shape)

    use_perennial_usage = np.percentile(perennial_usage_score, perenial_score_thres) > 0

    # Calculating hvac ts level confidence, by combining all scores

    if use_perennial_usage:
        weight1 = conf_score_weights_for_perenial_usage_user[0]
        weight2 = conf_score_weights_for_perenial_usage_user[1]
        weight3 = conf_score_weights_for_perenial_usage_user[2]
        weight4 = conf_score_weights_for_perenial_usage_user[3]

        detection_potential = weight1 * (hvac_usage_pot / np.max(hvac_usage_pot)) + \
                              weight2 * weekend_usage_score + \
                              weight3 * strikes_score + \
                              weight4 * perennial_usage_score
    else:
        weight1 = conf_score_weights[0]
        weight2 = conf_score_weights[1]
        weight3 = conf_score_weights[2]

        detection_potential = weight1 * (hvac_usage_pot / np.max(hvac_usage_pot)) + \
                              weight2 * weekend_usage_score + \
                              weight3 * strikes_score

    detection_potential[hvac_output == 0] = 0

    if use_perennial_usage:
        detection_potential[np.logical_and(perennial_usage_score < thres_for_perennial_cons_score_adjusment, detection_potential > 0)] = \
            np.minimum(detection_potential[np.logical_and(perennial_usage_score < thres_for_perennial_cons_score_adjusment, detection_potential > 0)], 0.2)

    # Calculating hvac ts level usage potential

    if np.all(hvac_usage_pot == 1):
        cool_estimation_potential = ((cooling_disagg + cool_ao) / cool_cap)
        heat_estimation_potential = ((heating_disagg + heat_ao) / heat_cap)

    else:
        weight1 = pot_score_weights[0]
        weight2 = pot_score_weights[1]

        cool_estimation_potential = weight1 * (hvac_usage_pot/np.max(hvac_usage_pot)) + weight2 * ((cooling_disagg + cool_ao) / cool_cap)
        heat_estimation_potential = weight1 * (hvac_usage_pot/np.max(hvac_usage_pot)) + weight2 * ((heating_disagg + heat_ao) / heat_cap)

    cool_estimation_potential[(cooling_disagg + cool_ao) == 0] = 0
    heat_estimation_potential[(heating_disagg + heat_ao) == 0] = 0

    cooling_usage_frac = (cooling_disagg + cool_ao) / cool_cap
    heating_usage_frac = (heating_disagg + heat_ao) / heat_cap

    cool_estimation_potential[cooling_usage_frac < thres_for_low_hvac_points_adjusment] = \
        np.minimum((cooling_usage_frac)[cooling_usage_frac < thres_for_low_hvac_points_adjusment],
                   cool_estimation_potential[cooling_usage_frac < thres_for_low_hvac_points_adjusment])

    heat_estimation_potential[heating_usage_frac < thres_for_low_hvac_points_adjusment] = \
        np.minimum((heating_usage_frac)[heating_usage_frac < thres_for_low_hvac_points_adjusment],
                   heat_estimation_potential[heating_usage_frac< thres_for_low_hvac_points_adjusment])

    cool_estimation_potential[perennial_usage_score < thres_for_perennial_cons_score_adjusment] = \
        np.minimum(cool_estimation_potential[perennial_usage_score < thres_for_perennial_cons_score_adjusment], 0.6)
    heat_estimation_potential[perennial_usage_score < thres_for_perennial_cons_score_adjusment] = \
        np.minimum(heat_estimation_potential[perennial_usage_score < thres_for_perennial_cons_score_adjusment], 0.6)

    cool_estimation_potential = np.fmax(0, cool_estimation_potential)
    heat_estimation_potential = np.fmax(0, heat_estimation_potential)
    detection_potential = np.fmax(0, detection_potential)
    cool_estimation_potential = np.fmin(1, cool_estimation_potential)
    heat_estimation_potential = np.fmin(1, heat_estimation_potential)
    detection_potential = np.fmin(1, detection_potential)

    cool_estimation_potential[vacation] = 0
    heat_estimation_potential[vacation] = 0
    detection_potential[vacation] = 0

    cool_detection_potential = copy.deepcopy(detection_potential)
    heat_detection_potential = copy.deepcopy(detection_potential)

    cooling_days = np.sum(cooling_disagg + cool_ao, axis=1) > 0
    heating_days = np.sum(heating_disagg + heat_ao, axis=1) > 0

    cool_detection_potential[heating_days] = 0
    cool_estimation_potential[heating_days] = 0

    heat_detection_potential[cooling_days] = 0
    heat_estimation_potential[cooling_days] = 0

    cool_detection_potential = np.fmin(1, cool_detection_potential)
    cool_estimation_potential = np.fmin(1, cool_estimation_potential)
    heat_detection_potential = np.fmin(1, heat_detection_potential)
    heat_estimation_potential = np.fmin(1, heat_estimation_potential)

    item_output_object["app_confidence"][cooling_index, :, :] = np.fmin(1, cool_detection_potential + cooling_score_offset)
    item_output_object["app_potential"][cooling_index, :, :] = np.fmin(1, cool_estimation_potential + cooling_score_offset)

    item_output_object["app_confidence"][heating_index, :, :] = np.fmin(1, heat_detection_potential)
    item_output_object["app_potential"][heating_index, :, :] = np.fmin(1, heat_estimation_potential)

    t_end = datetime.now()

    logger.info("HVAC potential calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object


def postprocess_hvac_components(hvac_output, usage_potential, weekend_usage_score, perennial_usage_score, seasonal_usage_score, strikes_score):

    """
    processing of individual score values that will be used for calculation of final ts level hvac potential

    Parameters:
        hvac_output               (np.ndarray)      : hvac disagg output
        usage_potential           (np.ndarray)      : HVAC usage score based on user behavior
        weekend_usage_score       (np.ndarray)      : HVAC usage score based on weekend/weekday behavior
        perennial_usage_score     (np.ndarray)      : HVAC usage score based on perennial load
        seasonal_usage_score      (np.ndarray)      : HVAC usage score based on seasonality
        strikes_score             (np.ndarray)      : HVAC usage score based on box type activity

    Returns:
        usage_potential           (np.ndarray)      : HVAC usage score based on user behavior
        weekend_usage_score       (np.ndarray)      : HVAC usage score based on weekend/weekday behavior
        perennial_usage_score     (np.ndarray)      : HVAC usage score based on perennial load
        seasonal_usage_score      (np.ndarray)      : HVAC usage score based on seasonality
        strikes_score             (np.ndarray)      : HVAC usage score based on box type activity
    """

    usage_potential = np.nan_to_num(usage_potential)
    weekend_usage_score = np.nan_to_num(weekend_usage_score)
    perennial_usage_score = np.nan_to_num(perennial_usage_score)
    seasonal_usage_score = np.nan_to_num(seasonal_usage_score)
    strikes_score = np.nan_to_num(strikes_score)

    usage_potential = np.fmax(0, usage_potential)
    weekend_usage_score = np.fmax(0, weekend_usage_score)
    perennial_usage_score = np.fmax(0, perennial_usage_score)
    seasonal_usage_score = np.fmax(0, seasonal_usage_score)
    strikes_score = np.fmax(0, strikes_score)

    usage_potential[hvac_output == 0] = 0
    weekend_usage_score[hvac_output == 0] = 0
    perennial_usage_score[hvac_output == 0] = 0
    seasonal_usage_score[hvac_output == 0] = 0
    strikes_score[hvac_output == 0] = 0

    strikes_score = np.array([0.1, 0.4, 0.6, 1])[np.digitize(strikes_score, [0.3, 0.5, 0.7], True)]

    return usage_potential, weekend_usage_score, perennial_usage_score, seasonal_usage_score, strikes_score


def get_weekday_profile(hvac_output, dow):

    """
    Calculate weekend/weekday energy profile

    Parameters:
        hvac_output            (np.ndarray)          : hvac disagg output
        dow                    (np.ndarray)          : DOW tags for all the days

    Returns:
        weekend_profile        (np.ndarray)          : weekend energy profile
        weekday_profile        (np.ndarray)          : weekday energy profile
    """

    length = len(hvac_output)

    days_in_weekday = 5
    days_in_weekend = 2

    weekend_profile = np.zeros(hvac_output.shape)
    weekday_profile = np.zeros(hvac_output.shape)

    weekend_days = np.logical_or(dow == 1, dow == 7)

    if np.any(weekend_days):
        weekend_start = np.where(weekend_days)[0][0]
        weekend_days[:weekend_start] = 1
        weekend_start = np.arange(weekend_start, length, 7)
    else:
        weekend_start = []

    if np.any(np.logical_not(weekend_days)):
        weekday_start = np.where(np.logical_not(weekend_days))[0][0]
        weekday_start = np.arange(weekday_start, length, 7)
    else:
        weekday_start = []

    for i, start in enumerate(weekday_start):
        weekday_profile[i] = np.mean(hvac_output[start: start+days_in_weekday], axis=0)

    for i, start in enumerate(weekend_start):
        weekend_profile[i] = np.mean(hvac_output[start: start+days_in_weekend], axis=0)

    weekday_profile = weekday_profile[~np.all(weekday_profile == 0, axis=1)]

    if not len(weekday_profile):
        return np.zeros(hvac_output.shape[1]), np.zeros(hvac_output.shape[1])

    weekday_profile = np.median(weekday_profile, axis=0)

    weekend_profile = weekend_profile[~np.all(weekend_profile == 0, axis=1)]

    if not len(weekend_profile):
        return np.zeros(hvac_output.shape[1]), np.zeros(hvac_output.shape[1])

    weekend_profile = np.median(weekend_profile, axis=0)

    return weekday_profile, weekend_profile


def get_seasonal_profile(original_input_data, season, window):

    """
    calculate winter, transition and summer energy profile

    Parameters:
        original_input_data     (np.ndarray)    : input data
        season                  (np.ndarray)    : season list
        window                  (int)           : length of window

    Returns:
        summer_profile          (np.ndarray)    : summer profile
        transition_profile      (np.ndarray)    : transition profile
        winter_profile          (np.ndarray)    : winter profile
    """

    winter_profile = np.zeros(original_input_data.shape)
    summer_profile = np.zeros(original_input_data.shape)
    transition_profile = np.zeros(original_input_data.shape)

    days_in_a_week = Cgbdisagg.DAYS_IN_WEEK

    percentile_used_for_season_profile = 70

    length = len(season)

    for i in range(0, length-window, 5):
        if np.sum(season[i: i + window]) > days_in_a_week:
            summer_profile[i] = np.percentile(original_input_data[i: i+window], percentile_used_for_season_profile, axis=0)
        if np.sum(season[i: i + window]) < -days_in_a_week:
            winter_profile[i] = np.percentile(original_input_data[i: i+window], percentile_used_for_season_profile, axis=0)
        if np.sum(season[i: i + window]) == 0:
            transition_profile[i] = np.percentile(original_input_data[i: i+window], percentile_used_for_season_profile, axis=0)

    if np.any(season == 0):
        if np.all(transition_profile == 0):
            transition_profile = np.percentile(original_input_data[season == 0], 25, axis=0)
        else:
            transition_profile = transition_profile[~np.all(transition_profile == 0, axis=1)]
            transition_profile = np.percentile(transition_profile, 25, axis=0)

    else:
        transition_profile = np.zeros(len(original_input_data[0]))

    return summer_profile, transition_profile, winter_profile
