"""
Author - Prasoon Patidar
Date - 08th June 2020
Lifestyle Submodule to calculate annual level lifestyle profile
"""

# import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.utils.logs_utils import log_prefix
from python3.analytics.lifestyle.functions.lifestyle_utils import get_yearly_load_type
from python3.analytics.lifestyle.functions.lifestyle_utils import get_consumption_level

from python3.utils.time.get_time_diff import get_time_diff
from python3.analytics.lifestyle.functions.office_goer_module import get_office_goer_annual_prob
from python3.analytics.lifestyle.functions.active_user_module import get_active_user_annual_prob
from python3.analytics.lifestyle.functions.weekend_warrior_module import get_weekend_warrior_annual_prob


def run_lifestyle_annual_module(lifestyle_input_object, lifestyle_output_object, logger_pass):

    """
    Parameters:
        lifestyle_input_object (dict)              : Dictionary containing all inputs for lifestyle modules
        lifestyle_output_object(dict)              : Dictionary containing all outputs for lifestyle modules
        logger_pass(dict)                          : Contains base logger and logging dictionary

    Returns:
        lifestyle_output_object(dict)              : Dictionary containing all outputs for lifestyle modules
    """

    t_lifestyle_annual_module_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('lifestyle_annual_module')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Lifestyle Annual attributes", log_prefix('Generic'))

    # Initialize annual dict in lifestyle output object

    lifestyle_output_object['annual'] = dict()

    # get input_data, clusters, day indices for this user run

    input_data = lifestyle_input_object.get('input_data')
    day_input_data = lifestyle_input_object.get('day_input_data')
    day_input_idx = lifestyle_input_object.get('day_input_data_index')

    # get season fractions for annual input data

    seasons = lifestyle_input_object.get('weather_config').get('season')

    season_fraction = np.zeros(seasons.count.value)

    day_season_fractions = lifestyle_input_object.get('day_input_data_seasons')

    season_val_count = np.bincount(day_season_fractions.astype(int))

    season_fraction[:season_val_count.shape[0]] = season_val_count / day_season_fractions.shape[0]

    logger.debug("%s Season Fractions: %s", log_prefix('Generic'), str(season_fraction))

    # write season fraction in lifestyle output object

    lifestyle_output_object['annual']['season_fraction'] = season_fraction

    # ------------------- SEASONAL CLUSTER SUB MODULE-------------------#

    kmeans_yearly_model = lifestyle_input_object.get('yearly_profile_kmeans_model')

    yearly_load_types = kmeans_yearly_model.get('cluster_labels')

    lifestyle_hsm = lifestyle_input_object.get('lifestyle_hsm')

    yearly_load_type, yearly_load_debug = \
        get_yearly_load_type(lifestyle_input_object, day_input_data, day_input_idx, kmeans_yearly_model, yearly_load_types, lifestyle_hsm, logger_pass)

    logger.info("%s yearly load type: %s", log_prefix('SeasonalLoadType'), yearly_load_type)

    if np.any(np.array(lifestyle_input_object.get('yearly_load_type_list')) == yearly_load_debug.get('yearly_load_type')):
        yearly_load_debug['yearly_load_cluster_id'] = \
            np.where(np.array(lifestyle_input_object.get('yearly_load_type_list')) == yearly_load_debug.get('yearly_load_type'))[0][0]
    else:
        yearly_load_debug['yearly_load_cluster_id'] = -1

    lifestyle_output_object['annual']['yearly_load_type'] = yearly_load_type

    lifestyle_output_object['annual']['yearly_load_debug'] = yearly_load_debug

    # ------------------- CONSUMPTION LEVEL SUB MODULE-------------------#

    pilot_based_config = lifestyle_input_object.get('pilot_based_config')

    consumption_levels = lifestyle_input_object.get('consumption_level')

    consumption_level, consumption_level_buckets, consumption_value = \
        get_consumption_level(input_data, pilot_based_config, consumption_levels, lifestyle_hsm, logger_pass, annual_tag=1)

    logger.info("%s Annual consumption level: %s", log_prefix('SeasonalLoadType'), consumption_level)

    lifestyle_output_object['annual']['consumption_level'] = consumption_level
    lifestyle_output_object['annual']['consumption_level_buckets'] = consumption_level_buckets
    lifestyle_output_object['annual']['consumption_value'] = consumption_value

    # ------------------- OFFICE GOER SUB MODULE-------------------#

    # Get overall office_goer status and probability at annual level

    is_office_goer, office_goer_prob, office_seasonal_prob = get_office_goer_annual_prob(lifestyle_input_object,
                                                                                         lifestyle_output_object,
                                                                                         logger_pass)
    logger.info("%s Annual office goer probability: %s", log_prefix('OfficeGoer'), str(office_goer_prob))

    lifestyle_output_object['annual']['is_office_goer'] = is_office_goer

    lifestyle_output_object['annual']['office_goer_prob'] = office_goer_prob

    lifestyle_output_object['annual']['office_goer_seasonal_prob'] = office_seasonal_prob

    # ------------------- ACTIVE USER SUB MODULE-------------------#

    # Get overall active_user status and probability at annual level

    is_active_user, active_user_prob, active_seasonal_prob =\
        get_active_user_annual_prob(lifestyle_input_object, lifestyle_output_object, logger_pass)

    logger.info("%s Annual active user probability: %s", log_prefix('ActiveUser'), str(active_user_prob))

    lifestyle_output_object['annual']['is_active_user'] = is_active_user

    lifestyle_output_object['annual']['active_user_prob'] = active_user_prob

    lifestyle_output_object['annual']['active_user_seasonal_prob'] = active_seasonal_prob

    # ------------------- WEEKEND WARRIOR SUB MODULE-------------------#

    # Get overall weekend_warrior status and probability at annual level

    is_weekend_warrior, weekend_warrior_prob, weekend_seasonal_prob = \
        get_weekend_warrior_annual_prob(lifestyle_input_object, lifestyle_output_object, logger_pass)

    logger.info("%s Annual weekend warrior probability: %s", log_prefix('WeekendWarrior'), str(weekend_warrior_prob))

    lifestyle_output_object['annual']['is_weekend_warrior'] = is_weekend_warrior

    lifestyle_output_object['annual']['weekend_warrior_prob'] = weekend_warrior_prob

    lifestyle_output_object['annual']['weekend_warrior_seasonal_prob'] = weekend_seasonal_prob

    t_lifestyle_annual_module_end = datetime.now()

    logger.info("%s Running lifestyle annual module | %.3f s", log_prefix('Generic'),
                get_time_diff(t_lifestyle_annual_module_start,
                              t_lifestyle_annual_module_end))

    return lifestyle_output_object
