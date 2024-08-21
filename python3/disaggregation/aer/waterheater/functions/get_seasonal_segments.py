"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to find season of each bill cycle / month
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.functions.timed.functions.box_features import Boxes
from python3.disaggregation.aer.waterheater.functions.get_seasons_cutoff import get_seasons_cutoff
from python3.disaggregation.aer.waterheater.functions.water_heater_utils import bill_cycle_to_month
from python3.disaggregation.aer.waterheater.functions.water_heater_utils import find_missing_bill_cycle_season


# List of columns and their corresponding indices used in seasonal info

season_columns = {
    'bill_cycle_ts': 0,
    'average_temp': 1,
    'season_id': 2,
    'thin_monthly': 3,
    'fat_monthly': 4,
    'raw_monthly': 5,
    'thin_of_total': 6,
    'thin_of_wh': 7,
    'fat_of_total': 8,
    'fat_of_wh': 9,
    'num_days': 10,
}


def get_seasonal_segments(in_data, timed_boxes, debug, logger_base, wh_config, monthly=True, return_data=True,
                          one_season=False, setpoint=65):
    """
    Parameters:
        in_data         (np.ndarray)    : Input 21-column matrix
        timed_boxes     (list)          : Timed water heater boxes features
        debug           (dict)          : Algorithm intermediate steps output
        logger_base     (dict)          : Logger object
        wh_config       (dict)          : Config parameters for water heater
        monthly         (bool)          : Flag for monthly analysis
        return_data     (bool)          : Flag for returning season data
        one_season      (bool)          : Flag for treating whole data as one season
        setpoint        (int)           : Default setpoint to find seasons

    Returns:
        wtr_tuple       (tuple)         : Winter data and indices
        itr_tuple       (tuple)         : Transition data and indices
        smr_tuple       (tuple)         : Summer data and indices
        all_seasons     (np.ndarray)    : Seasonal information
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_seasonal_segments')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking deepcopy of input data to keep local instance

    input_data = deepcopy(in_data)

    # Retrieve season id mapping

    if timed_boxes is not None:
        season_code = wh_config['season_code']
    else:
        season_code = wh_config['timed_wh']['season_code']

    # Bill cycle timestamp to month timestamp

    if monthly:
        # Bill cycle timestamps conversion to actual month timestamps

        input_data = bill_cycle_to_month(input_data)

        logger.info('Bill cycles converted to monthly | ')
    else:
        logger.info('Bill cycles not converted to monthly | ')

    # Consider only non-NAN temperatures

    temp_data = deepcopy(input_data)
    temp_data = temp_data[~np.isnan(temp_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX])]

    # Get unique bill cycle timestamps and corresponding indices

    unique_months, months_idx, months_count = np.unique(temp_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                                                        return_counts=True,
                                                        return_inverse=True)

    # Calculate monthly average temperature

    avg_temp = np.bincount(months_idx, temp_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]) / months_count
    avg_temp = np.round(avg_temp, 3)

    # Stack bill cycle timestamps and temperature together

    seasons_info = np.hstack((unique_months.reshape(-1, 1), np.round(avg_temp.reshape(-1, 1), 3),
                              np.empty((len(avg_temp), 1))))

    # Find season for bill cycle with no valid temperature value

    seasons_info = find_missing_bill_cycle_season(input_data, seasons_info, logger)

    # Sort the season data based on average monthly temperature

    seasons_info = seasons_info[np.lexsort((seasons_info[:, season_columns['bill_cycle_ts']],
                                            seasons_info[:, season_columns['average_temp']]))]

    # Check difference of average temperature with respect to set point

    temp_diff = np.abs(seasons_info[:, season_columns['average_temp']] - setpoint)

    # Get the allowed number of transition months from config

    n_transitions = wh_config['num_transition_months']

    # Find the transition month index

    if len(temp_diff) > n_transitions:
        # If number of months more than required number of transition months

        transition_limit = sorted(temp_diff)[n_transitions]
    else:
        # If number of months less than required number of transition months

        transition_limit = setpoint

    # Retrieve the cutoff value's index

    transition_idx = np.where(temp_diff <= transition_limit)[0]

    # Get the temperature bounds for winter and summer

    wtr_cutoff, smr_cutoff = get_seasons_cutoff(seasons_info, season_columns, transition_idx, unique_months, one_season)

    # Separating winter season segment

    wtr_temp_seg = seasons_info[seasons_info[:, season_columns['average_temp']] < wtr_cutoff, :]
    wtr_temp_seg[:, season_columns['season_id']] = season_code['wtr']

    # Separating intermediate season segment

    itr_temp_seg = seasons_info[(seasons_info[:, season_columns['average_temp']] >= wtr_cutoff) &
                                (seasons_info[:, season_columns['average_temp']] <= smr_cutoff), :]
    itr_temp_seg[:, season_columns['season_id']] = season_code['itr']

    # Separating summer season segment

    smr_temp_seg = seasons_info[seasons_info[:, season_columns['average_temp']] > smr_cutoff, :]
    smr_temp_seg[:, season_columns['season_id']] = season_code['smr']

    # Combining all season segments and sort by bill cycle timestamp

    all_seasons = np.vstack((wtr_temp_seg, itr_temp_seg, smr_temp_seg))
    all_seasons = all_seasons[all_seasons[:, season_columns['bill_cycle_ts']].argsort()]

    # Separating indices for each season in the input data

    wtr_idx = np.in1d(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], wtr_temp_seg[:, season_columns['bill_cycle_ts']])
    itr_idx = np.in1d(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], itr_temp_seg[:, season_columns['bill_cycle_ts']])
    smr_idx = np.in1d(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], smr_temp_seg[:, season_columns['bill_cycle_ts']])

    # Check whether timed water heater and if data is to be returned

    if timed_boxes is not None:
        # Getting seasonal output for timed water heater

        # Retrieve boxes features

        boxes, col = timed_boxes

        # Initialize the seasons id array for each epoch in input data

        season_array = np.zeros(input_data.shape[0])

        # Assign season id to each corresponding season index

        season_array[wtr_idx] = season_code['wtr']
        season_array[itr_idx] = season_code['itr']
        season_array[smr_idx] = season_code['smr']

        # Get day number from the input data

        unq_days, days_idx = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)

        # Update the season and day number for all the boxes

        boxes[:, Boxes.SEASON] = season_array[boxes[:, col].astype(int)]
        boxes[:, Boxes.DAY_NUMBER] = days_idx[boxes[:, col].astype(int)]

        # Updating debug object with relevant indices

        debug['variables']['wtr_idx'] = wtr_idx
        debug['variables']['itr_idx'] = itr_idx
        debug['variables']['smr_idx'] = smr_idx

        return (wtr_temp_seg, itr_temp_seg, smr_temp_seg), debug, boxes

    # Getting seasonal output for thermostat water heater

    if return_data:
        # If data to be returned, separate the input data for each season

        wtr_data = input_data[wtr_idx, :]
        itr_data = input_data[itr_idx, :]
        smr_data = input_data[smr_idx, :]

        # Create tuple for each seasons (season info, season data)

        wtr_tuple = (wtr_temp_seg, wtr_data)
        itr_tuple = (itr_temp_seg, itr_data)
        smr_tuple = (smr_temp_seg, smr_data)

        logger.info('Seasons info returned along with data | ')

        return wtr_tuple, itr_tuple, smr_tuple, all_seasons

    else:
        logger.info('Seasons info returned without data | ')

        return all_seasons
