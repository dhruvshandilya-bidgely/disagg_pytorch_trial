"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update WH confidence and potential values
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.pilot_constants import PilotConstants

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.raw_energy_itemization.utils import get_boxes
from python3.itemization.aer.raw_energy_itemization.utils import get_box_score
from python3.itemization.aer.raw_energy_itemization.utils import postprocess_conf_arr

from python3.itemization.aer.raw_energy_itemization.appliance_potential.config.get_pot_conf import get_pot_conf


def get_wh_potential(app_index, item_input_object, item_output_object, sampling_rate, vacation, thin_pulse, logger_pass):

    """
    Calculate WH confidence and potential values

    Parameters:
        app_index                   (int)           : Index of app in the appliance list
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        sampling_rate               (int)           : sampling rate
        vacation                    (np.ndarray)    : array of vacation days
        thin_pulse                  (np.ndarray)    : Thin pulse consumption
        logger_pass                 (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)          : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_wh_potential')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Fetching required inputs

    wh_disagg = item_output_object.get("updated_output_data")[app_index, :, :]

    config = get_pot_conf().get('wh')

    perc_used_to_get_max_val = config.get('perc_used_to_get_max_val')
    conf_score_of_thin_pulse_cons = config.get('conf_score_of_thin_pulse_cons')
    min_wh_conf_score = config.get('min_wh_conf_score')
    min_seasonal_potential = config.get('min_seasonal_potential')

    # removing thin pulse component of WH before calculating ts level potential/confidence value
    # confidence of thin pulse points will be added later

    if thin_pulse is not None:
        wh_disagg = wh_disagg - thin_pulse

    wh_disagg = np.fmax(0, wh_disagg)

    if np.all(wh_disagg == 0):
        return item_output_object

    seasonal_potential = item_output_object.get("wh_pot")[:, 0]

    item_input_object['seasonal_potential'] = seasonal_potential

    activity_curve = item_output_object.get('debug').get("profile_attributes_dict").get("activity_curve")

    wh_box_features = get_boxes(wh_disagg, int(Cgbdisagg.SEC_IN_HOUR/sampling_rate))

    # Calculate wh confidence using consistency in amplitude and start time

    consistent_usage_score, start_usage_score = get_box_score(wh_disagg, wh_box_features)

    start_usage_score = np.fmin(1, start_usage_score + 0.1)

    wh_potential = (np.fmin(1, wh_disagg/np.percentile(wh_disagg, perc_used_to_get_max_val))*1)

    wh_potential[wh_disagg == 0] = 0

    # initializing WH confidence based on consistency and time of use of WH fat pulse boxes

    wh_confidence = config.get("start_score_weight") * start_usage_score + \
                    config.get("consistent_usage_score_weight") * np.nan_to_num(consistent_usage_score)

    activity_curve = activity_curve/np.max(activity_curve) + config.get("act_curve_inc")

    wh_confidence = np.fmin(1, np.multiply(wh_confidence, activity_curve[None, :]))

    wh_confidence[start_usage_score > config.get("max_conf")] = \
        np.maximum(config.get("max_conf"), wh_confidence[start_usage_score > config.get("max_conf")])

    processed_season_potential = np.zeros_like(wh_confidence)
    processed_season_potential[:, :] = np.fmax(min_seasonal_potential, seasonal_potential)[:, None]

    # updating WH confidence based on weather analytics WH potential

    wh_confidence = np.multiply(wh_confidence, processed_season_potential)

    wh_potential = np.fmax(wh_potential, 0)

    wh_potential = wh_potential / np.percentile(wh_potential[wh_potential > 0], perc_used_to_get_max_val)

    wh_confidence, wh_potential = postprocess_conf_arr(wh_confidence, wh_potential, wh_disagg, vacation)

    wh_confidence[wh_disagg == 0] = 0

    wh_confidence[wh_disagg > 0] = np.fmax(min_wh_conf_score, wh_confidence[wh_disagg > 0])

    # adding WH potential and confidence for thin pulse boxes

    if thin_pulse is not None:
        thin_pulse_pot = thin_pulse / np.percentile(wh_disagg[wh_disagg > 0], perc_used_to_get_max_val)

        # Adding thin pulse potential in wh potential

        wh_potential = wh_potential + thin_pulse_pot
        wh_confidence[thin_pulse > 0] = conf_score_of_thin_pulse_cons

    # Dumping appliance confidence and potential values

    pilot = item_input_object.get('config').get('pilot_id')


    if pilot in PilotConstants.SEASONAL_WH_ENABLED_PILOTS:
        wh_confidence[:, :] = 1

    item_output_object["app_confidence"][app_index, :, :] = wh_confidence
    item_output_object["app_potential"][app_index, :, :] = wh_potential

    t_end = datetime.now()

    logger.info("WH potential calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object
