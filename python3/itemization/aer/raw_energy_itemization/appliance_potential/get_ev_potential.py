
"""
Author - Nisha Agarwal
Date - 10th Mar 2021
Calculate ev ts level confidence and potential values
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.itemization_utils import resample_day_data

from python3.itemization.aer.raw_energy_itemization.utils import get_boxes
from python3.itemization.aer.raw_energy_itemization.utils import get_box_score
from python3.itemization.aer.raw_energy_itemization.utils import postprocess_conf_arr

from python3.itemization.aer.raw_energy_itemization.appliance_potential.config.get_pot_conf import get_pot_conf


def get_ev_potential(app_index, item_input_object, item_output_object, sampling_rate, weather_analytics, vacation, logger_pass):

    """
    Calculate EV confidence and potential values

    Parameters:
        app_index                   (int)           : Index of app in the appliance list
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        sampling_rate               (int)           : sampling rate
        weather_analytics           (dict)          : weather analytics module output
        vacation                    (np.ndarray)    : array of vacation days
        logger_pass                 (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)          : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    ev_config = get_pot_conf().get("ev")

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_ev_potential')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Fetching required inputs

    ev_disagg = item_output_object.get("updated_output_data")[app_index, :, :]

    if np.all(ev_disagg == 0):
        return item_output_object

    # Calculate ev confidence using consistency in amplitude and start time

    ev_box_features = get_boxes(ev_disagg, int(Cgbdisagg.SEC_IN_HOUR/sampling_rate))

    consistent_usage_score, start_usage_score = get_box_score(ev_disagg, ev_box_features)

    season_potential = get_season_potential(ev_disagg, sampling_rate)

    # calculating EV timestamp level confidence values , using three factors -
    # 1, start time consistency,
    # 2, box location consistency,

    ev_confidence = (start_usage_score + consistent_usage_score) / 2

    ev_l1_tag_threshold = ev_config.get('ev_l1_tag_threshold')
    season_potential_score_bucket = ev_config.get('season_potential_score_bucket')
    consistent_usage_score_bucket = ev_config.get('consistent_usage_score_bucket')
    min_conf_for_disagg_users = ev_config.get('min_conf_for_disagg_users')

    if (item_input_object.get('created_hsm') is not None) and (item_input_object.get('created_hsm').get('ev') is not None) and \
            (item_input_object.get('created_hsm').get('ev').get('attributes'))\
            and (item_input_object.get('created_hsm').get('ev').get('attributes').get("ev_amplitude") is not None):

        logger.info("ev amplitude info is available in ev hsm | ")

        amp = item_input_object.get('created_hsm').get('ev').get('attributes').get("ev_amplitude")

        type = item_input_object.get('created_hsm').get('ev').get('attributes').get("charger_type")

        if type is not None:

            logger.info("ev type info is available in ev hsm | ")

            type = type == 1

            if type:

                logger.info("disagg ev charger type is l1 | ")

                # Assign score to each L1 box , this additional scoring is only present for EV l1 user

                ev_confidence_copy = get_l1_box_score(ev_box_features, ev_disagg, ev_config)

                ev_confidence = copy.deepcopy(ev_confidence_copy)
        else:

            if amp < ev_l1_tag_threshold:

                logger.info("ev charger type is L1 | ")

                # Assign score to each L1 box , this additional scoring is only present for EV l1 user

                ev_confidence_copy = get_l1_box_score(ev_box_features, ev_disagg, ev_config)

                ev_confidence = copy.deepcopy(ev_confidence_copy)

    # Reducing confidence scores of EV boxes, that overlap at HVAC region
    # This helps to reduce or eliminate FP EV boxes

    low_cosistency_and_high_seasonal_points = np.logical_and(consistent_usage_score < consistent_usage_score_bucket[0],
                                                             season_potential < season_potential_score_bucket[0])

    ev_confidence[low_cosistency_and_high_seasonal_points] = \
        np.fmin(season_potential_score_bucket[0], ev_confidence[low_cosistency_and_high_seasonal_points])

    low_cosistency_and_high_seasonal_points = np.logical_and(consistent_usage_score < consistent_usage_score_bucket[1],
                                                             season_potential < season_potential_score_bucket[1])

    ev_confidence[low_cosistency_and_high_seasonal_points] = np.fmin(season_potential_score_bucket[1],
                                                                     ev_confidence[low_cosistency_and_high_seasonal_points])

    # Calculate EV TS level potential
    ev_potential = (ev_disagg / np.percentile(ev_disagg[ev_disagg > 0], 98))

    ev_confidence[ev_disagg == 0] = 0
    ev_potential[ev_disagg == 0] = 0

    if (item_input_object.get('created_hsm') is not None) and (item_input_object.get('created_hsm').get('ev') is not None) \
            and (item_input_object.get('created_hsm').get('ev').get('attributes').get("ev_probability") is not None):

        prob = min(1, 0.4 + item_input_object.get('created_hsm').get('ev').get('attributes').get("ev_probability"))

        if prob < 0.7:
            ev_confidence = ev_confidence * (prob + 0.1)
            ev_confidence = np.fmin(1, ev_confidence)

        logger.info("EV disagg confidence score is | %s",  item_input_object.get('created_hsm').get('ev').get('attributes').get("ev_probability"))

    ev_confidence[np.logical_and(ev_disagg > 0, ev_confidence <= min_conf_for_disagg_users)] = min_conf_for_disagg_users

    ev_confidence, ev_potential = postprocess_conf_arr(ev_confidence, ev_potential, ev_disagg, vacation.astype(bool))

    # Dumping appliance confidence and potential values

    item_output_object["app_confidence"][app_index, :, :] = ev_confidence
    item_output_object["app_potential"][app_index, :, :] = ev_potential

    t_end = datetime.now()

    logger.debug("EV potential calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object


def get_l1_box_score(box_feature, ev_disagg, ev_config):

    """
    Assign confidence to each EV L1 box

    Parameters:
        box_feature             (np.ndarray)        : features of each ev l1 box
        ev_disagg               (np.ndarray)        : ev l1 disagg output
        ev_config               (dict)              : ev config dictionary

    Returns:
        ev_confidence           (np.ndarray)        : confidence of each ev l1 box
    """

    config = get_pot_conf().get("ev")
    min_ev_conf = config.get('min_ev_conf')

    samples_per_hour = int(ev_disagg.shape[1] / Cgbdisagg.HRS_IN_DAY)

    start_score_arr = ev_config.get('start_score_arr')
    end_score_arr = ev_config.get('end_score_arr')
    length_score_arr = ev_config.get('length_score_arr')
    amp_score_arr = ev_config.get('amp_score_arr')

    start_score_arr = np.array(start_score_arr)
    end_score_arr = np.array(end_score_arr)
    length_score_arr = np.array(length_score_arr)
    amp_score_arr = np.array(amp_score_arr)

    start_score = start_score_arr[(box_feature[:, 2] / samples_per_hour).astype(int)]
    end_score = end_score_arr[(box_feature[:, 3] / samples_per_hour).astype(int)]
    length_score = length_score_arr[(box_feature[:, 4] / samples_per_hour).astype(int)]

    amp_score = np.ones_like(box_feature[:, 10]) * 0.5

    valid_idx = (box_feature[:, 10] / (100/samples_per_hour)) < 35

    amp_score[valid_idx] = amp_score_arr[((box_feature[:, 10] / (100/samples_per_hour))[[valid_idx]]).astype(int)]

    cons_tou_score = np.sum(ev_disagg > 0, axis=0) / np.sum(ev_disagg.sum(axis=1) > 0)

    cons_tou_score_1d = np.zeros_like(ev_disagg)
    cons_tou_score_1d[:, :] = cons_tou_score[None, :]

    start = box_feature[:, 7].astype(int)
    end = box_feature[:, 8].astype(int)

    start_score_1d = np.zeros(ev_disagg.size)
    end_score_1d = np.zeros(ev_disagg.size)
    length_score_1d = np.zeros(ev_disagg.size)
    amp_score_1d = np.zeros(ev_disagg.size)

    for idx in range(len(box_feature)):
        start_score_1d[start[idx]: end[idx] + 1] = start_score[idx]
        end_score_1d[start[idx]: end[idx] + 1] = end_score[idx]
        length_score_1d[start[idx]: end[idx] + 1] = length_score[idx]
        amp_score_1d[start[idx]: end[idx] + 1] = amp_score[idx]

    start_score_1d = np.reshape(start_score_1d, ev_disagg.shape)
    end_score_1d = np.reshape(end_score_1d, ev_disagg.shape)
    amp_score_1d = np.reshape(amp_score_1d, ev_disagg.shape)
    length_score_1d = np.reshape(length_score_1d, ev_disagg.shape)

    ev_confidence = (start_score_1d + end_score_1d + cons_tou_score_1d + amp_score_1d + length_score_1d) / 5

    ev_confidence[ev_disagg == 0] = 0
    ev_confidence[ev_confidence < min_ev_conf] = 0

    ev_confidence = np.fmax(0, ev_confidence)
    ev_confidence = np.fmin(1, ev_confidence)

    return ev_confidence


def get_season_potential(input_data, sampling_rate):

    """
    Get hourly seasonal potential

    Parameters:
        weather_analytics           (dict)            : weather analytics information dict
        input_data                  (np.ndarray)      : input raw data
        sampling_rate               (int)             : number of samples in an hour

    Returns:
        season_potential            (np.ndarray)      : calculated season potential
    """

    season_potential = np.zeros(input_data.shape)

    season_potential = np.nan_to_num(season_potential)

    season_potential = resample_day_data(season_potential, int(Cgbdisagg.SEC_IN_HOUR / sampling_rate) * Cgbdisagg.HRS_IN_DAY)

    season_potential = np.fmin(1, 1 - season_potential + 0.1)

    return season_potential
