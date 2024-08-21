"""
Author - Prasoon Patidar
Date - 08th June 2020
Lifestyle Submodule to calculate seasonal level lifestyle profile
"""

# import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.logs_utils import log_prefix
from python3.utils.time.get_time_diff import get_time_diff
from python3.analytics.lifestyle.functions.get_cluster_info import get_cluster_fractions
from python3.analytics.lifestyle.functions.lowcooling_module import get_lowcooling_constant
from python3.analytics.lifestyle.functions.active_user_module import get_active_user_probability
from python3.analytics.lifestyle.functions.office_goer_module import get_office_goer_probability
from python3.analytics.lifestyle.functions.lifestyle_utils import get_consumption_level
from python3.analytics.lifestyle.functions.weekend_warrior_module import get_weekend_warrior_probability


def run_lifestyle_season_module(lifestyle_input_object, lifestyle_output_object, logger_pass):

    """
    Parameters:
        lifestyle_input_object (dict)              : Dictionary containing all inputs for lifestyle modules
        lifestyle_output_object(dict)              : Dictionary containing all outputs for lifestyle modules
        logger_pass(dict)                          : Contains base logger and logging dictionary

    Returns:
        lifestyle_output_object(dict)              : Dictionary containing all outputs for lifestyle modules
    """

    t_lifestyle_season_module_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('lifestyle_seasonal_module')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Lifestyle Seasonal attributes", log_prefix('Generic'))

    # get input_data, clusters, day indices and season for this user run

    seasons = lifestyle_input_object.get('weather_config').get('season')
    input_data = lifestyle_input_object.get('input_data')
    day_input_idx = lifestyle_input_object.get('day_input_data_index')
    day_clusters = lifestyle_output_object.get('day_clusters')
    daily_load_type = lifestyle_input_object.get('daily_load_type')

    # Get 2D data from lifestyle input object

    day_input_data = lifestyle_input_object.get('day_input_data')

    # Get lowcooling info for office goer users

    lowcooling_constant, lowcooling_debug_info = get_lowcooling_constant(lifestyle_input_object, logger_pass)

    # write lowcooling constant in lifestyle output object

    lifestyle_output_object['lowcooling_constant'] = lowcooling_constant

    lifestyle_output_object['lowcooling_debug'] = lowcooling_debug_info

    # loop over seasons for getting profile for each out bill cycle

    lifestyle_output_object['season'] = dict()

    season_idx = lifestyle_input_object.get('SEASON_IDX')

    weekday_idx = lifestyle_input_object.get('WEEKDAY_IDX')

    for season_id in range(seasons.count.value):

        season_name = seasons(season_id).name

        logger.debug("%s Processing Season | %s", log_prefix('Generic'), season_name)

        # Create empty array for season

        lifestyle_output_object['season'][season_name] = dict()

        # split input data, clusters and day_index for this season

        season_input_data = input_data[input_data[:, season_idx] == season_id, :]

        logger.debug("%s Season %s Input Data Shape | %s", log_prefix('Generic'), season_name, str(season_input_data.shape))

        # return if season input data has zero rows

        if season_input_data.shape[0]==0:

            logger.info("%s No input data present for season %s. season not processed.",
                        log_prefix('Generic'), seasons(season_id).name)

            continue

        # ----------------------- CLUSTERING SUB MODULE-----------------------

        # get cluster fractions based on season input data

        cluster_fractions = get_cluster_fractions(season_input_data, day_clusters, day_input_idx, daily_load_type,
                                                  logger_pass)

        logger.debug("%s Season %s Input Cluster Fractions | %s", log_prefix('SeasonalLoadType'), season_name, str(cluster_fractions))

        # Write cluster fractions in output object

        lifestyle_output_object['season'][season_name]['cluster_fraction'] = cluster_fractions

        # split season input data for weekdays in this season

        weekday_season_input_data = season_input_data[season_input_data[:, weekday_idx] == 1, :]

        # get cluster fractions based on season input data

        weekday_cluster_fractions = get_cluster_fractions(weekday_season_input_data, day_clusters, day_input_idx,
                                                          daily_load_type,
                                                          logger_pass)

        logger.debug("%s Season %s Input Cluster Fractions Weekday | %s",
                     log_prefix('SeasonalLoadType'), season_name, str(weekday_cluster_fractions))

        # Write cluster fractions in output object

        lifestyle_output_object['season'][season_name]['weekday_cluster_fraction'] = weekday_cluster_fractions

        # split season input data for weekends in this season

        weekend_season_input_data = season_input_data[season_input_data[:, weekday_idx] == 0, :]

        # get cluster fractions based on season input data

        weekend_cluster_fractions = get_cluster_fractions(weekend_season_input_data, day_clusters, day_input_idx,
                                                          daily_load_type,
                                                          logger_pass)

        logger.debug("%s Season %s Input Cluster Fractions Weekend | %s",
                     log_prefix('SeasonalLoadType'), season_name, str(weekend_cluster_fractions))

        # Write cluster fractions in output object

        lifestyle_output_object['season'][season_name]['weekend_cluster_fraction'] = weekend_cluster_fractions

        # ------------------- CONSUMPTION LEVEL SUB MODULE-------------------#

        pilot_based_config = lifestyle_input_object.get('pilot_based_config')

        consumption_levels = lifestyle_input_object.get('consumption_level')

        lifestyle_hsm = lifestyle_input_object.get('lifestyle_hsm')

        consumption_level_season, _, _ = get_consumption_level(season_input_data, pilot_based_config, consumption_levels, lifestyle_hsm, logger_pass, annual_tag=0)

        logger.debug("%s Season %s Consumption Level | %s",
                     log_prefix('SeasonalLoadType'), season_name, str(consumption_level_season))

        lifestyle_output_object['season'][season_name]['consumption_level'] = consumption_level_season


        # ----------------------- WEEKEND WARRIOR SUB MODULE-----------------------

        # Get Weekend Warrior Seasonal Probabilities

        weekend_warrior_seasonal_prob, weekend_warrior_debug_info = get_weekend_warrior_probability(season_input_data,
                                                                                                    lifestyle_input_object,
                                                                                                    logger_pass)

        logger.debug("%s Season %s | Weekend Warrior prob | %s",
                     log_prefix('WeekendWarrior'), season_name, str(weekend_warrior_seasonal_prob))

        # write weekend warrior probability in lifestyle output object

        if weekend_warrior_seasonal_prob is not None:

            lifestyle_output_object['season'][season_name]['weekend_warrior_prob'] = weekend_warrior_seasonal_prob

            lifestyle_output_object['season'][season_name]['weekend_warrior_debug'] = weekend_warrior_debug_info

        # -------------------------- ACTIVE USER SUB MODULE--------------------------

        # get weights config for active user prob based on season

        active_user_config = lifestyle_input_object.get('active_user_config')

        active_user_weights = active_user_config.get(season_name)

        # get day input data for this season

        season_day_vals = np.unique(season_input_data[:, Cgbdisagg.INPUT_DAY_IDX])

        season_day_idx = np.isin(day_input_idx, season_day_vals)

        # Get Day idx, day data and day clusters for this bill cycle

        day_input_data_season = day_input_data[season_day_idx]

        # get active user seasonal probabilities

        active_user_seasonal_prob, active_user_debug_info = get_active_user_probability(day_input_data_season,
                                                                                        active_user_weights,
                                                                                        lifestyle_input_object,
                                                                                        logger_pass)

        logger.debug("%s Season %s Active user prob | %s",
                     log_prefix('ActiveUser'), season_name, str(active_user_seasonal_prob))

        if active_user_seasonal_prob is not None:

            lifestyle_output_object['season'][season_name]['active_user_prob'] = active_user_seasonal_prob

            lifestyle_output_object['season'][season_name]['active_user_debug'] = active_user_debug_info

        # -------------------------- OFFICE GOER SUB MODULE--------------------------

        # get day input data for this season

        season_weekday_vals = np.unique(weekday_season_input_data[:,Cgbdisagg.INPUT_DAY_IDX])

        season_weekday_idx = np.isin(day_input_idx, season_weekday_vals)

        # Get Day idx, day data and day clusters for this bill cycle

        weekday_input_idx_season = day_input_idx[season_weekday_idx]

        weekday_input_data_season = day_input_data[season_weekday_idx]

        weekday_clusters = day_clusters[season_weekday_idx]

        # Get office goer probabilities

        office_goer_seasonal_prob, office_goer_debug_info = get_office_goer_probability(weekday_input_data_season,
                                                                                        weekday_input_idx_season,
                                                                                        weekday_clusters,
                                                                                        weekday_cluster_fractions,
                                                                                        lowcooling_constant,
                                                                                        lifestyle_input_object,
                                                                                        logger_pass)

        logger.debug("%s Season %s Office goer prob | %s",
                     log_prefix('OfficeGoer'), season_name, str(office_goer_seasonal_prob))

        # write office goer probability in lifestyle output object

        if office_goer_seasonal_prob is not None:

            lifestyle_output_object['season'][season_name]['office_goer_prob'] = office_goer_seasonal_prob

            lifestyle_output_object['season'][season_name]['office_goer_debug'] = office_goer_debug_info

    t_lifestyle_season_module_end = datetime.now()

    logger.info("%s Running lifestyle seasonal module | %.3f s", log_prefix('Generic'),
                get_time_diff(t_lifestyle_season_module_start,
                              t_lifestyle_season_module_end))

    return lifestyle_output_object
