"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to get the non-timed thermostat water heater features at bill cycle / monthly level
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.disaggregation.aer.waterheater.functions.thermostat.functions.get_moving_laps import get_moving_laps
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.get_consolidated_laps import get_consolidated_laps
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.get_bill_cycle_features import get_bill_cycle_features


def get_seasonal_features(winter, intermediate, summer, wh_config, debug, logger_base):
    """
    Paramaters:
        winter              (tuple)         : Data and indices of winter season
        intermediate        (tuple)         : Data and indices of the transition season
        summer              (tuple)         : Data and indices of the summer season
        wh_config           (dict)          : Config params
        debug               (dict)          : Algorithm intermediate steps output
        logger_base         (logger)        : logger object

    Returns:
        all_features        (np.ndarray)    : Features for each bill cycle / month
        debug               (dict)          : Algorithm intermediate steps output
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_seasonal_features')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking a deepcopy of input data to keep local instances

    winter_tuple = deepcopy(winter)
    intermediate_tuple = deepcopy(intermediate)
    summer_tuple = deepcopy(summer)

    # Extracting seasonal data and indices from the tuple

    wtr_month, wtr_data = winter_tuple
    itr_month, itr_data = intermediate_tuple
    smr_month, smr_data = summer_tuple

    # #---------------------------------------------- Winter season -------------------------------------------------- #

    # Get moving laps for the season

    wtr_lap_mid_timestamps, wtr_peaks_idx, wtr_laps_idx = get_moving_laps(wtr_data, wh_config, logger)

    # Combine the laps

    wtr_consolidated_laps, wtr_data = get_consolidated_laps(wtr_data, wtr_lap_mid_timestamps, wtr_laps_idx, wh_config)

    # Get bill_cycle / monthly features for the season

    wtr_features, wtr_lap_peaks = get_bill_cycle_features(wtr_data, wtr_consolidated_laps, wtr_month, wtr_peaks_idx,
                                                          wh_config, logger)

    # #--------------------------------------------- Intermediate season --------------------------------------------- #

    # Get moving laps for the season

    itr_lap_mid_timestamps, itr_peaks_idx, itr_laps_idx = get_moving_laps(itr_data, wh_config, logger)

    # Combine the laps

    itr_consolidated_laps, itr_data = get_consolidated_laps(itr_data, itr_lap_mid_timestamps, itr_laps_idx, wh_config)

    # Get bill_cycle / monthly features for the season

    itr_features, itr_lap_peaks = get_bill_cycle_features(itr_data, itr_consolidated_laps, itr_month, itr_peaks_idx,
                                                          wh_config, logger)

    # #---------------------------------------------- Summer season -------------------------------------------------- #

    # Get moving laps for the season

    smr_lap_mid_timestamps, smr_peaks_idx, smr_laps_idx = get_moving_laps(smr_data, wh_config, logger)

    # Combine the laps

    smr_consolidated_laps, smr_data = get_consolidated_laps(smr_data, smr_lap_mid_timestamps, smr_laps_idx, wh_config)

    # Get bill_cycle / monthly features for the season

    smr_features, smr_lap_peaks = get_bill_cycle_features(smr_data, smr_consolidated_laps, smr_month, smr_peaks_idx,
                                                          wh_config, logger)

    # #--------------------------------- Combining all the seasonal features ------------------------------------------#

    all_features = np.vstack((wtr_features, itr_features, smr_features))

    # Cache the interim data to debug object with the following features

    # data          : Raw data of the season
    # laps          : The laps of the season
    # peaks         : Peaks index of the season
    # features      : Detection features of the season
    # lap_peaks     : Lap peaks index of the season

    debug['season_features'] = {
        'wtr': {
            'data': wtr_data,
            'laps': wtr_consolidated_laps,
            'peaks': wtr_peaks_idx,
            'features': wtr_features,
            'lap_peaks': wtr_lap_peaks
        },
        'itr': {
            'data': itr_data,
            'laps': itr_consolidated_laps,
            'peaks': itr_peaks_idx,
            'features': itr_features,
            'lap_peaks': itr_lap_peaks
        },
        'smr': {
            'data': smr_data,
            'laps': smr_consolidated_laps,
            'peaks': smr_peaks_idx,
            'features': smr_features,
            'lap_peaks': smr_lap_peaks
        }
    }

    return all_features, debug
