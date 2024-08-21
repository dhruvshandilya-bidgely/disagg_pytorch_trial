"""
Author - Prasoon Patidar
Date - 08th June 2020
Lifestyle Submodule to calculate event level lifestyle profile
"""

# import python packages

import logging
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.logs_utils import log_prefix
from python3.utils.time.get_time_diff import get_time_diff
from python3.analytics.lifestyle.functions.get_cluster_info import get_cluster_fractions


def run_lifestyle_event_module(lifestyle_input_object, lifestyle_output_object, logger_pass):

    """
    Parameters:
        lifestyle_input_object (dict)              : Dictionary containing all inputs for lifestyle modules
        lifestyle_output_object(dict)              : Dictionary containing all outputs for lifestyle modules
        logger_pass(dict)                          : Contains base logger and logging dictionary

    Returns:
        lifestyle_output_object(dict)              : Dictionary containing all outputs for lifestyle modules
    """

    t_lifestyle_event_module_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('lifestyle_event_module')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Lifestyle Event attributes", log_prefix('Generic'))

    # get input_data, clusters, day indices and outbill cycles for this user run

    input_data = lifestyle_input_object.get('input_data')
    day_cluster_idx = lifestyle_input_object.get('day_input_data_index')
    day_clusters = lifestyle_output_object.get('day_clusters')
    daily_load_type = lifestyle_input_object.get('daily_load_type')

    # Get event start and end times from gb disagg event

    event_start_time = lifestyle_input_object.get('event_start_time')
    event_end_time = lifestyle_input_object.get('event_end_time')

    logger.debug("%s Event start time:%d, end time:%d", log_prefix('Generic'), event_start_time, event_end_time)

    # Trim input data based on event star and end time

    event_input_data = input_data[input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] > event_start_time, :]

    event_input_data = event_input_data[event_input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] <= event_end_time, :]

    logger.debug("%s Event Input Data Shape | %s", log_prefix('Generic'), str(event_input_data.shape))

    # create empty dict and get indices for event level profile in output object

    lifestyle_output_object['event'] = dict()

    WEEKDAY_IDX = lifestyle_input_object.get('WEEKDAY_IDX')

    # get cluster fractions based on event input data

    cluster_fractions = get_cluster_fractions(event_input_data, day_clusters, day_cluster_idx, daily_load_type,
                                              logger_pass)

    logger.debug("%s Event Input Cluster Fractions | %s", log_prefix('Generic'), str(cluster_fractions))

    # Write cluster fractions in output object

    lifestyle_output_object['event']['cluster_fraction'] = cluster_fractions

    # split event input data for weekdays in this event time

    weekday_event_input_data = event_input_data[event_input_data[:, WEEKDAY_IDX] == 1, :]

    # get cluster fractions based on event input data

    weekday_cluster_fractions = get_cluster_fractions(weekday_event_input_data, day_clusters, day_cluster_idx,
                                                      daily_load_type,
                                                      logger_pass)

    logger.debug("%s Event Input Cluster Fractions Weekday | %s",
                 log_prefix('Generic'), str(weekday_cluster_fractions))

    # Write cluster fractions in output object

    lifestyle_output_object['event']['weekday_cluster_fraction'] = weekday_cluster_fractions

    # split event input data for weekends in this event time

    weekend_event_input_data = event_input_data[event_input_data[:, WEEKDAY_IDX] == 0, :]

    # get cluster fractions based on event input data

    weekend_cluster_fractions = get_cluster_fractions(weekend_event_input_data, day_clusters, day_cluster_idx,
                                                      daily_load_type,
                                                      logger_pass)

    logger.debug("%s Event Input Cluster Fractions Weekend | %s",
                 log_prefix('Generic'), str(weekend_cluster_fractions))

    # Write cluster fractions in output object

    lifestyle_output_object['event']['weekend_cluster_fraction'] = weekend_cluster_fractions

    t_lifestyle_event_module_end = datetime.now()

    logger.info("%s Running lifestyle event module | %.3f s", log_prefix('Generic'),
                get_time_diff(t_lifestyle_event_module_start,
                              t_lifestyle_event_module_end))

    return lifestyle_output_object
