"""
Author - Prasoon Patidar
Date - 08th June 2020
Lifestyle Submodule to calculate bill-cycle level lifestyle profile
"""

# import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.logs_utils import log_prefix
from python3.utils.time.get_time_diff import get_time_diff
from python3.analytics.lifestyle.functions.get_peaks_info import get_peaks_info
from python3.analytics.lifestyle.functions.get_cluster_info import get_cluster_fractions

from python3.analytics.lifestyle.functions.lifestyle_utils import get_hour_fraction
from python3.analytics.lifestyle.functions.lifestyle_utils import get_consumption_level

from python3.analytics.lifestyle.functions.wakeup_sleep_module import get_sleep_attributes
from python3.analytics.lifestyle.functions.wakeup_sleep_module import get_wakeup_attributes


def run_lifestyle_bc_module(lifestyle_input_object, lifestyle_output_object, logger_pass):

    """
    Parameters:
        lifestyle_input_object (dict)              : Dictionary containing all inputs for lifestyle modules
        lifestyle_output_object(dict)              : Dictionary containing all outputs for lifestyle modules
        logger_pass(dict)                          : Contains base logger and logging dictionary

    Returns:
        lifestyle_output_object(dict)              : Dictionary containing all outputs for lifestyle modules
    """

    t_lifestyle_bc_module_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('lifestyle_bill_cycle_module')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Lifestyle Bill Cycle level attributes", log_prefix('Generic'))

    # get input_data, clusters, day indices, vacation info and outbill cycles for this user run

    out_bill_cycles = lifestyle_input_object.get('out_bill_cycles')

    input_data = lifestyle_input_object.get('input_data')

    day_cluster_idx = lifestyle_input_object.get('day_input_data_index')

    day_clusters = lifestyle_output_object.get('day_clusters')

    daily_load_type = lifestyle_input_object.get('daily_load_type')

    # Get 2D data from lifestyle input object

    day_input_idx = lifestyle_input_object.get('day_input_data_index')

    day_input_data = lifestyle_input_object.get('day_input_data')

    # loop over all out bill cycles for getting profile for each out bill cycle

    BILLCYCLE_START_COL = 0

    WEEKDAY_IDX = lifestyle_input_object.get('WEEKDAY_IDX')

    # remove any out bill cycles which are older than first bill cycle in lifestyle input data

    min_bill_cycle_val = np.min(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX])

    out_bill_cycles = out_bill_cycles[out_bill_cycles[:, BILLCYCLE_START_COL] >= min_bill_cycle_val]

    lifestyle_output_object['billcycle'] = dict()

    for billcycle_start in out_bill_cycles[:, BILLCYCLE_START_COL]:

        logger.debug("%s Processing Billcycle | %s", log_prefix('Generic'), str(billcycle_start))

        # Create empty arr with billcycle start

        lifestyle_output_object['billcycle'][billcycle_start] = dict()

        # ---------------------- CLUSTERING SUB MODULE----------------------#

        # split input data, clusters and day_index for this bill cycle

        billcycle_input_data = input_data[input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == billcycle_start, :]

        logger.debug("%s Billcycle %s Input Data Shape | %s",
                     log_prefix('Generic'), str(billcycle_start), str(billcycle_input_data.shape))

        # return if billcycle input data has zero rows

        if billcycle_input_data.shape[0] == 0:
            logger.info("%s No input data present for billcycle %d. billcycle not processed.",
                        log_prefix('Generic'), billcycle_start)

            continue

        # get cluster fractions based on billcycle input data

        cluster_fractions = get_cluster_fractions(billcycle_input_data, day_clusters, day_cluster_idx, daily_load_type,
                                                  logger_pass)

        logger.debug("%s Billcycle %s Input Cluster Fractions | %s",
                     log_prefix('Generic'), str(billcycle_start), str(cluster_fractions))

        # Write cluster fractions in output object

        lifestyle_output_object['billcycle'][billcycle_start]['cluster_fraction'] = cluster_fractions

        # split billcycle input data for weekdays in this bill cycle

        weekday_billcycle_input_data = billcycle_input_data[billcycle_input_data[:, WEEKDAY_IDX] == 1, :]

        logger.debug("%s Billcycle %s Weekday Input Shape | %s",
                     log_prefix('Generic'), str(billcycle_start), str(weekday_billcycle_input_data.shape))

        # get cluster fractions based on billcycle input data

        weekday_cluster_fractions = get_cluster_fractions(weekday_billcycle_input_data, day_clusters, day_cluster_idx,
                                                          daily_load_type,
                                                          logger_pass)

        logger.debug("%s Billcycle %s Weekday Cluster Fractions | %s",
                     log_prefix('Generic'), str(billcycle_start), str(weekday_cluster_fractions))

        # Write cluster fractions in output object

        lifestyle_output_object['billcycle'][billcycle_start]['weekday_cluster_fraction'] = weekday_cluster_fractions

        # split billcycle input data for weekends in this bill cycle

        weekend_billcycle_input_data = billcycle_input_data[billcycle_input_data[:, WEEKDAY_IDX] == 0, :]

        logger.debug("%s Billcycle %s Weekend Input Shape | %s",
                     log_prefix('Generic'), str(billcycle_start), str(weekend_billcycle_input_data.shape))

        # get cluster fractions based on billcycle input data

        weekend_cluster_fractions = get_cluster_fractions(weekend_billcycle_input_data, day_clusters, day_cluster_idx,
                                                          daily_load_type,
                                                          logger_pass)

        logger.debug("%s Billcycle %s Weekend Cluster Fractions | %s",
                     log_prefix('DailyLoadType'), str(billcycle_start), str(weekend_cluster_fractions))

        # Write cluster fractions in output object

        lifestyle_output_object['billcycle'][billcycle_start]['weekend_cluster_fraction'] = weekend_cluster_fractions

        # ------------------- HOUR Fraction SUB MODULE-------------------#

        hour_fraction = get_hour_fraction(billcycle_input_data, day_input_data, day_input_idx)

        lifestyle_output_object['billcycle'][billcycle_start]['hour_fraction'] = hour_fraction

        logger.debug("%s Billcycle %s Hour Fraction | %s",
                     log_prefix('HourFractions'), str(billcycle_start), str(hour_fraction))

        weekday_hour_fraction = get_hour_fraction(weekday_billcycle_input_data, day_input_data, day_input_idx)

        lifestyle_output_object['billcycle'][billcycle_start]['weekday_hour_fraction'] = weekday_hour_fraction

        logger.debug("%s Billcycle %s Weekday Hour Fraction | %s",
                     log_prefix('HourFractions'), str(billcycle_start), str(weekday_hour_fraction))

        weekend_hour_fraction = get_hour_fraction(weekend_billcycle_input_data, day_input_data, day_input_idx)

        lifestyle_output_object['billcycle'][billcycle_start]['weekend_hour_fraction'] = weekend_hour_fraction

        logger.debug("%s Billcycle {str(billcycle_start)} Weekend Hour Fraction | %s",
                     log_prefix('HourFractions'), str(weekend_hour_fraction))

        # ------------------- CONSUMPTION LEVEL SUB MODULE-------------------#

        pilot_based_config = lifestyle_input_object.get('pilot_based_config')

        consumption_levels = lifestyle_input_object.get('consumption_level')

        lifestyle_hsm = lifestyle_input_object.get('lifestyle_hsm')

        consumption_level_bill_cycle, _, _ = \
            get_consumption_level(billcycle_input_data, pilot_based_config, consumption_levels, lifestyle_hsm, logger_pass, annual_tag=0)


        logger.debug("%s Billcycle %s Consumption Level | %s",
                     log_prefix('DailyLoadType'), str(billcycle_start), str(consumption_level_bill_cycle))

        lifestyle_output_object['billcycle'][billcycle_start]['consumption_level'] = consumption_level_bill_cycle

        # ---------------------- Wakeup/Sleep SUB MODULE----------------------#

        # get lighting TOU and wakeup sleep config for wake up/ sleep time detection

        lighting_time_of_usage = lifestyle_input_object.get('lighting_hourly_bands')

        wakeup_sleep_config = lifestyle_input_object.get('wakeup_sleep_config')

        if np.sum(lighting_time_of_usage) <= 0:

            logger.info("%s Unable to get lighting Bands from HSM, skipping detection of wakeup/sleep hours",
                        log_prefix(['WakeUpTime','SleepTime'], type='list'))

        else:

            # Get wakeup attributes for this bill cycle

            wakeup_attributes = get_wakeup_attributes(lifestyle_input_object, billcycle_input_data, lighting_time_of_usage, wakeup_sleep_config,
                                                      logger_pass)

            # Get sleep attributes for this bill cycle

            sleep_attributes = get_sleep_attributes(lifestyle_input_object, billcycle_input_data, lighting_time_of_usage, wakeup_sleep_config,
                                                    logger_pass)

            # write wakeup sleep attributes in output object

            lifestyle_output_object['billcycle'][billcycle_start]['wakeup'] = wakeup_attributes

            lifestyle_output_object['billcycle'][billcycle_start]['sleep'] = sleep_attributes

        # ---------------------- Peak Detection SUB MODULE----------------------#

        # get start and end day for this bill cycle

        bill_cycle_start_day = billcycle_input_data[0, Cgbdisagg.INPUT_DAY_IDX]

        bill_cycle_end_day = billcycle_input_data[-1, Cgbdisagg.INPUT_DAY_IDX]

        # Get Day idx, day data and day clusters for this bill cycle

        day_input_idx_bill_cycle = \
            day_input_idx[(day_input_idx >= bill_cycle_start_day) & (day_input_idx <= bill_cycle_end_day)]

        day_input_data_bill_cycle = \
            day_input_data[(day_input_idx >= bill_cycle_start_day) & (day_input_idx <= bill_cycle_end_day)]

        day_clusters_bill_cycle = np.array([day_clusters[day_cluster_idx == day][0]
                                            for day in day_input_idx_bill_cycle])

        # get consumption peaks information for this bill cycle

        peaks_info_bill_cycle = get_peaks_info(day_input_data_bill_cycle,
                                               day_input_idx_bill_cycle,
                                               day_clusters_bill_cycle,
                                               cluster_fractions,
                                               lifestyle_input_object,
                                               logger_pass)

        # write peaks info for this bill cycle

        lifestyle_output_object['billcycle'][billcycle_start]['peaks'] = peaks_info_bill_cycle

    t_lifestyle_bc_module_end = datetime.now()

    logger.info("%s Running lifestyle billcycle module | %.3f s", log_prefix('Generic'),
                get_time_diff(t_lifestyle_bc_module_start,
                              t_lifestyle_bc_module_end))

    return lifestyle_output_object
