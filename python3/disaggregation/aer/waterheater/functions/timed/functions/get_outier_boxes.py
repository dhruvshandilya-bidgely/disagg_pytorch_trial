"""
Author - Nikhil
Date - 10/10/2018
Module to filter the outlier boxes within each season
"""

# Import python packages

import logging
import numpy as np
from scipy import stats
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.functions.timed.functions.box_features import Boxes


def get_outlier_boxes(box_info, max_hours, box_data, debug, logger_base):
    """
    Removing the outlier boxes

    Parameters:
        box_info        (np.ndarray)    : Boxes features
        max_hours       (np.ndarray)    : High energy time divisions
        box_data        (np.ndarray)    : Input 21-column box data
        debug           (dict)          : Algorithm intermediate steps output
        logger_base     (dict)          : Logger object
    Returns:
        season_boxes    (np.ndarray)    : Updated season boxes
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('seasonal_noise_filter')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking deepcopy of boxes features to make local instances

    season_boxes = deepcopy(box_info)

    # Get water heater type (start / end)

    wh_type = debug['wh_type']

    # Get sampling rate factor

    factor = debug['time_factor']
    n_divisions = Cgbdisagg.HRS_IN_DAY * factor

    # Check if any max hours outside the permitted range

    max_hours = np.r_[(max_hours - factor), max_hours, (max_hours + factor)]

    # Fixing corner cases for max hours (< 0, > n_divisions)

    max_hours[max_hours < 0] += n_divisions
    max_hours[max_hours > (n_divisions - 1)] -= n_divisions

    # Take unique max hours

    max_hours = np.unique(max_hours)

    # Initialise the bool for marking invalid boxes

    max_boxes = np.array([True] * season_boxes.shape[0])

    # Iterate over each time division

    for hour in max_hours:
        # If the boxes edge match with significant hour, mask it valid

        max_boxes[(season_boxes[:, Boxes.TIME_DIVISION] == hour)] = False

    # Column mapping with water heater type

    col = 0 if wh_type == 'start' else 1

    # Get median of seasonal boxes energy data

    season_boxes_energy = box_data[season_boxes[:, col].astype(int), Cgbdisagg.INPUT_CONSUMPTION_IDX]
    median_energy = np.median(season_boxes_energy)

    # Calculate the acceptable minimum and maximum percentile with respect to median

    min_percentile = stats.percentileofscore(season_boxes_energy, 0.5 * median_energy)
    max_percentile = stats.percentileofscore(season_boxes_energy, 2.0 * median_energy)

    logger.debug('Min and max percentile for box energy: | {}, {}'.format(min_percentile, max_percentile))

    # Calculate the corresponding min and max energy allowed

    energy_min = np.percentile(season_boxes_energy, min_percentile)
    energy_max = np.percentile(season_boxes_energy, max_percentile)

    logger.debug('Min and max box energy allowed outside important hours: | {}, {}'.format(energy_min, energy_max))

    # Mark boxes that violate the energy bounds

    invalid_boxes = (season_boxes_energy < energy_min) | (season_boxes_energy > energy_max)

    # Remove boxes if they don't overlap with significant hours

    invalid_boxes = invalid_boxes & max_boxes

    logger.debug('Number of boxes removed due to invalid energy amplitude: | {}'.format(np.sum(invalid_boxes)))

    # Update the boxes features

    season_boxes[invalid_boxes, Boxes.IS_VALID] = 0

    return season_boxes
