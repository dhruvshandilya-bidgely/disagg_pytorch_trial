
"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Process hvac potential and season data
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def process_weather_analytics_output(item_input_object, item_output_object):

    """
    Process hvac potential and season data

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs

    Returns:
        weather_analytics         (dict)      : dict containing weather info
        season                    (np.ndarray): season tag list
        item_output_object        (dict)      : Dict containing all outputs
    """

    cooling_pot = item_input_object.get('item_input_params').get('input_data')[Cgbdisagg.INPUT_COOLING_POTENTIAL_IDX]
    heating_pot = item_input_object.get('item_input_params').get('input_data')[Cgbdisagg.INPUT_HEATING_POTENTIAL_IDX]
    wh_pot = item_input_object.get('item_input_params').get('input_data')[Cgbdisagg.INPUT_WH_POTENTIAL_IDX]
    season = item_input_object.get('item_input_params').get('input_data')[Cgbdisagg.INPUT_S_LABEL_IDX]

    samples_per_hour = int(item_input_object.get('item_input_params').get('input_data').shape[2] / Cgbdisagg.HRS_IN_DAY)

    # fetch hvac potential values from weather analytics output

    if np.any(np.sum(cooling_pot, axis=0) > 0):
        index = (np.where(np.sum(cooling_pot, axis=0) > 0)[0][0]) % samples_per_hour
    else:
        index = 0

    cooling_pot = cooling_pot[:, np.arange(index, samples_per_hour * Cgbdisagg.HRS_IN_DAY, samples_per_hour)]
    heating_pot = heating_pot[:, np.arange(index, samples_per_hour * Cgbdisagg.HRS_IN_DAY, samples_per_hour)]
    wh_pot = wh_pot[:, np.arange(index, samples_per_hour * Cgbdisagg.HRS_IN_DAY, samples_per_hour)]

    if np.any(np.sum(season != 0, axis=0) > 0):
        index = np.where(np.sum(season != 0, axis=0) > 0)[0][0]
    else:
        index = 0

    item_output_object["season"] = season[:, index]
    item_output_object["season_potential"] = cooling_pot + heating_pot
    item_output_object["heating_pot"] = heating_pot
    item_output_object["cooling_pot"] = cooling_pot
    item_output_object["wh_pot"] = wh_pot

    season = item_output_object["season"]

    weather_analytics = dict()
    weather_analytics["weather"] = dict()
    weather_analytics["weather"]["hvac_potential_dict"] = dict()
    weather_analytics["weather"]["season_detection_dict"] = dict()

    weather_analytics["weather"]["hvac_potential_dict"]["cooling_pot"] = cooling_pot
    weather_analytics["weather"]["hvac_potential_dict"]["heating_pot"] = heating_pot
    weather_analytics["weather"]["hvac_potential_dict"]["wh_pot"] = wh_pot
    weather_analytics["weather"]["season_detection_dict"]["s_label"] = season

    return weather_analytics, season, item_output_object
