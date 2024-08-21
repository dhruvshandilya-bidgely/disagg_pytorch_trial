
"""
Author - Nisha Agarwal
Date - 10th Mar 2021
Calculate li ts level confidence and potential values
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
from python3.itemization.aer.functions.itemization_utils import rolling_func_along_col

from python3.itemization.aer.raw_energy_itemization.appliance_potential.config.get_pot_conf import get_pot_conf


def get_li_potential(app_index, item_input_object, item_output_object, cloud_cover, sampling_rate, vacation, logger_pass):

    """
    Calculate Li confidence and potential values

    Parameters:
        app_index                   (int)           : Index of app in the appliance list
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        cloud_cover                 (np.ndarray)    : Cloud cover data
        sampling_rate               (int)           : sampling rate
        vacation                    (np.ndarray)    : vacation data
        logger_pass                 (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object          (dict)          : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_li_potential')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Fetching required inputs

    config = get_pot_conf().get("li")

    activity_curve = item_input_object.get("activity_curve")
    occupancy_profile = item_output_object.get("occupants_profile")
    sleep_hours = item_output_object.get("profile_attributes").get("sleep_hours")
    input_data = item_input_object.get("item_input_params").get("day_input_data")
    li_confidence = item_output_object.get('debug').get('lighting_module_dict').get("lighting_day_estimate_after_postprocess")

    if not np.all(li_confidence == 0):
        li_confidence = li_confidence / np.max(li_confidence)

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')
    # Lighting potential is already calculated in lighting module, adding mid day potential based on cloud cover data

    cloud_cover = rolling_func_along_col(cloud_cover, samples_per_hour)

    if item_input_object.get('item_input_params').get('run_hybrid_v2_flag') and \
            occupancy_profile.get("occupants_features")[occupancy_profile.get("stay_at_home_index")]:

        # adding day lighting only for stay at home users
        # otherwise lighting is only given before sunrise and after sunset

        logger.info("Adding mid day lighting for the user")

        cloud_cover = resample_day_data(cloud_cover, samples_per_hour * Cgbdisagg.HRS_IN_DAY)

        # adding day lighting where cloud cover is greater than certain threshold

        li_cloud_cover_potential = copy.deepcopy(cloud_cover)
        li_cloud_cover_potential[cloud_cover < config.get("min_cloud_cover")] = 0

        low_cloud_cover_points = np.logical_and(li_confidence > 0, cloud_cover < config.get("min_cloud_cover_in_active_hour"))

        li_cloud_cover_potential[low_cloud_cover_points] = 0

        normalized_input_data = (input_data - np.percentile(input_data, 3, axis=1)[:, None]) / \
                                (np.percentile(input_data, 97, axis=1) - np.percentile(input_data, 3, axis=1))[:, None]

        normalized_input_data = np.round(normalized_input_data, 2)

        normalized_input_data = np.fmax(normalized_input_data, 0)

        activity_curve_2d = np.zeros(normalized_input_data.shape)

        # adding day lighting based on activity trend of the user

        activity_curve_2d[:, :] = activity_curve

        usage_potential = np.minimum(normalized_input_data, activity_curve_2d)

        cloud_cover[:, np.logical_not(sleep_hours)] = 0

        li_cloud_cover_potential = np.multiply(usage_potential, cloud_cover)

        li_confidence = li_confidence + li_cloud_cover_potential
        li_confidence = np.fmin(li_confidence, 1)

        logger.info("Adding day time lighting for the user")

    li_potential = copy.deepcopy(li_confidence)

    # Final sanity checks

    li_confidence = np.ones(li_confidence.shape)

    li_potential[vacation] = 0
    li_confidence[vacation] = 0

    # Heatmap dumping section

    item_output_object["app_confidence"][app_index, :, :] = li_confidence
    item_output_object["app_potential"][app_index, :, :] = li_potential

    t_end = datetime.now()

    logger.info("LI potential calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object
