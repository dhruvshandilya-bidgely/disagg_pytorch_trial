"""
Author - Nikhil Singh Chauhan
Date - 16/10/18
This is the module to check if the total water heater consumption sufficient enough
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.waterheater.functions.get_seasonal_segments import season_columns


def check_final_consumption_percentage(debug, wh_config, logger_base):
    """
    Parameters:
        debug       (dict)      : The object to save all important data and values
        wh_config   (dict)      : Water heater config params
        logger_base (dict)      : Logger object

    Returns:
         debug      (dict)      : The object to save all important data and values
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('final_consumption_check')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve the required bounds from config

    detection_config = wh_config['thermostat_wh']['detection']

    wh_of_total_min = detection_config['wh_of_total_min']
    average_consumption = detection_config['average_consumption']
    consumption_probability = detection_config['consumption_probability']

    disagg_mode = wh_config.get('disagg_mode')

    # Retrieve the bill cycle consumption of water heater and detection probability

    if disagg_mode != 'mtd':
        # If historical/incremental mode, use detection probability

        detection_probability = debug['thermostat_hld_prob']
    else:
        # In case of mtd mode, use default probability 1

        detection_probability = 1

    wh_estimate = deepcopy(debug['wh_estimate'])
    final_estimate = deepcopy(debug['final_estimate'])

    # Retrieve the water heater and raw consumption from debug object

    input_consumption = final_estimate[:, season_columns['raw_monthly']]
    water_heater_consumption = wh_estimate[:, 1]

    total_input_consumption = np.nansum(input_consumption)
    total_water_heater_consumption = np.nansum(water_heater_consumption)

    # Calculate the water heater consumption as a percent of total consumption

    wh_of_total_percent = np.round(100 * total_water_heater_consumption / total_input_consumption, 4)

    average_water_heater_consumption = np.nanmean(water_heater_consumption)

    # Check which criteria of consumption and probability fail

    wh_of_total_fail = True if wh_of_total_percent < wh_of_total_min else False

    probability_fail = True if detection_probability < consumption_probability else False

    consumption_fail = True if average_water_heater_consumption < average_consumption else False

    # Check if any of the above three conditions failed

    overall_detection_fail = wh_of_total_fail & probability_fail & consumption_fail

    # Log all the checks and their corresponding status

    logger.info('Water heater of total percent vs threshold | {}, {}'.format(wh_of_total_percent, wh_of_total_min))

    logger.info('Average water heater consumption vs threshold | {}, {}'.format(average_water_heater_consumption,
                                                                                average_consumption))
    logger.info('Detection probability vs threshold | {}, {}'.format(detection_probability,
                                                                     consumption_probability))

    # Check if the checks sufficient enough to change detection status

    if (disagg_mode != 'mtd') and overall_detection_fail:
        # If consumption checks failed, reset detection

        debug['thermostat_hld'] = 0

        # Make water heater consumption zero and reassign to debug object

        wh_estimate[:, 1] = 0

        debug['wh_estimate'] = wh_estimate

        debug['final_wh_signal'][:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

        logger.info('Water heater detection made zero due to consumption check failure | ')
    else:
        logger.info('Water heater consumption check passed | ')

    # If thermostat hld is 1 but there is no estimation then set the thermostat hld to 0

    if (disagg_mode != 'mtd') and total_water_heater_consumption == 0 and debug['thermostat_hld'] == 1:
        debug['thermostat_hld'] = 0
        logger.info('Thermostat hld made zero due to 0 estimation | ')

    return debug
