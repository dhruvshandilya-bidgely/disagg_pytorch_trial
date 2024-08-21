"""
Author - Nisha Agarwal
Date - 10/9/20
Script for ref features calculation
"""

# Import python packages

import copy
import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants

from python3.itemization.aer.functions.itemization_utils import resample_day_data


def filter_appliance(consumption, ref_config):

    """
    Filter appliance day level data

    Parameters:
        consumption         (numpy.ndarray)     : Original consumption
        ref_config          (dict)              : Dictionary containing all config information

    Returns:
        consumption         (numpy.ndarray)     : Filtered consumption
    """

    hour_window = ref_config.get("filter_consumption").get("hour_window")
    day_window = ref_config.get("filter_consumption").get("day_window")
    hours_in_a_day = Cgbdisagg.HRS_IN_DAY

    # filter thin pulse consumption

    consumption[consumption < np.percentile(np.unique(consumption), ref_config.get("filter_consumption").get("percentile"))] = 0

    length = len(consumption)

    for i in range(0, length, day_window):
        count = np.count_nonzero(consumption[i: i+day_window, :], axis=0) < day_window/2
        consumption[i: i + day_window, count] = 0

    for i in range(0, hours_in_a_day, hour_window):
        count = np.count_nonzero(consumption[:, i:i+hour_window], axis=1) < hour_window/2
        consumption[count, i:i + hour_window] = 0

    return consumption


def prepare_raw_energy_features(item_input_object):

    """
    Prepare raw energy features for ref estimation

    Parameters:
        item_input_object     (dict)             : Dictionary containing all inputs

    Returns:
        features                (numpy.ndarray)    : Raw energy features for ref estimation
    """

    # Fetch daily raw energy data

    raw_energy = item_input_object.get("item_input_params").get("day_input_data")

    # Fill nan and negative values in raw data with 0

    raw_energy = np.nan_to_num(raw_energy)
    raw_energy = np.fmax(raw_energy, 0)

    # Calculate day level total energy

    day_level_energy = np.sum(raw_energy) / len(raw_energy)

    # Remove 0 values to calculate base load

    raw_energy[raw_energy == 0] = 1000000000

    # resample data to 3600

    downsampled_energy = resample_day_data(raw_energy, total_samples=Cgbdisagg.HRS_IN_DAY)

    # Calculate base load

    baseload = np.min(downsampled_energy)

    features = np.array([day_level_energy, baseload])

    return features


def prepare_cooling_features(item_input_object, ref_config):

    """
    Prepare cooling features for ref estimation

    Parameters:
        item_input_object     (dict)             : Dictionary containing all inputs
        ref_config            (dict)             : dict containing config values of ref module

    Returns:
        features                (numpy.ndarray)    : cooling features for ref estimation
    """

    # fetch day level cooling estimate

    daily_cooling = item_input_object.get("item_input_params").get("output_data")[
        item_input_object.get("item_input_params").get("app_list").index("cooling") + 1]

    # Resample to cooling estimate to 3600

    hourly_cooling = resample_day_data(daily_cooling, total_samples=Cgbdisagg.HRS_IN_DAY)

    # Calculate maximum cooling of the user

    cooling_level = np.percentile(hourly_cooling, ref_config.get("features").get("cooling_percentile"))

    # Filter cooling consumption

    filtered_cooling = filter_appliance(hourly_cooling, ref_config)

    # Calculate maximum hour of usage of cooling

    filtered_cooling2 = copy.deepcopy(filtered_cooling)
    filtered_cooling2[filtered_cooling2 <= np.percentile(filtered_cooling, 10, axis=1)[:, None]] = 0

    max_cooling_hours = 12

    cooling_hours = np.sum((filtered_cooling2 > 0).sum(axis=0) / np.max((filtered_cooling2 > 0).sum(axis=0)) > 0.7)

    features = np.array([cooling_level, min(max_cooling_hours, cooling_hours)])

    return features


def prepare_features(item_input_object, ref_config, logger):

    """
    Prepare features for ref estimation

    Parameters:
        item_input_object     (dict)             : Dictionary containing all inputs
        ref_config              (dict)             : Dictionary containing all config information
        logger                  (logger)           : logger object

    Returns:
        features                (numpy.ndarray)    : cooling features for ref estimation
        model_category          (str)              : Model category to be used for estimation
    """

    # Default model is used to predict ref output using only raw input data

    model_category = "raw"

    # Calculate raw energy features

    features = prepare_raw_energy_features(item_input_object)

    logger.debug("Prepared raw energy features")

    # Fetch cooling disagg output estimates

    cooling_column = item_input_object.get("disagg_output_write_idx_map").get("hvac")[0]

    cooling_disagg_estimate = item_input_object.get("disagg_epoch_estimate")[:, cooling_column]

    # Calculate cooling features, if cooling data is available

    pilot = item_input_object.get("config").get("pilot_id")

    if (not np.all(cooling_disagg_estimate == 0)) and pilot not in PilotConstants.HVAC_JAPAN_PILOTS:

        # Model to be used is using both raw energy and cooling

        model_category = model_category + "_hvac"

        cooling_features = prepare_cooling_features(item_input_object, ref_config)

        logger.debug("Prepared cooling features")

        features = np.append(features, cooling_features)

    # Fetch home meta features

    home_meta_data = item_input_object["home_meta_data"]

    living_area = home_meta_data.get("livingArea", 0)
    num_of_bedrooms = home_meta_data.get("bedrooms", 0)
    num_of_occupants = home_meta_data.get("numOccupants", 0)

    meta_features_present = living_area and num_of_occupants and num_of_bedrooms

    # If meta features are available, also use meta data to predict ref consumption

    if meta_features_present:

        living_area_limits = ref_config.get('meta_data').get("living_area_limits")
        num_of_bedrooms_limits = ref_config.get('meta_data').get("num_of_bedrooms_limits")
        num_of_occupants_limits = ref_config.get('meta_data').get("num_of_occupants_limits")

        if living_area < living_area_limits[1] and living_area > living_area_limits[0] and \
                num_of_bedrooms < num_of_bedrooms_limits[1] and num_of_bedrooms > num_of_bedrooms_limits[0] and \
                num_of_occupants < num_of_occupants_limits[1] and num_of_occupants > num_of_occupants_limits[0]:

            model_category = model_category + "_meta"

            meta_features = np.array([living_area, num_of_bedrooms, num_of_occupants])

            features = np.append(features, meta_features)

    logger.info("Prepared ref features | %s", features)
    logger.info("Model category to be used for ref estimate | %s", model_category)

    return features, model_category
