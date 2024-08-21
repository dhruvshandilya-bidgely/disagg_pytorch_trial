"""
Author - Prasoon Patidar
Date - 03rd June 2020
Populate lifestyle profile payload using lifestyle attributes
"""

# import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.logs_utils import log_prefix
from python3.utils.time.get_time_diff import get_time_diff
from python3.utils.validate_output_schema import validate_lifestyle_profile_schema_for_billcycle


def populate_lifestyle_user_profile(lifestyle_input_object, lifestyle_output_object, disagg_output_object, logger_pass):

    """
    Parameters:
        lifestyle_input_object(dict)            : Dictionary containing lifestyle specific inputs
        lifestyle_output_object(dict)           : Dictionary containing lifestyle specific outputs
        disagg_output_object(dict)              : Dictionary containing all disagg outputs
        logger_pass(dict)                       : Contains base logger and logging dictionary

    Returns:
        disagg_output_object(dict)              : Dictionary containing all disagg outputs
    """

    t_populate_lifestyle_profile_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('populate_lifestyle_user_profile')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Populating Lifestyle Profile", log_prefix('Generic'))

    # get required outputs for billcycle, season and annual level

    season_output = lifestyle_output_object.get('season')

    annual_output = lifestyle_output_object.get('annual')

    event_output = lifestyle_output_object.get('event')

    daily_load_type = lifestyle_input_object.get('daily_load_type')

    # get Dominant load type of the user

    dominant_load_type = daily_load_type(np.argmax(event_output.get('cluster_fraction'))).name \
        if (event_output.get('cluster_fraction') is not None) else None

    # get fraction of days of the Dominant load type

    dominant_load_fraction = float(max(event_output.get('cluster_fraction'))) \
        if (event_output.get('cluster_fraction') is not None) else None

    vacation_perc = 0.0

    # Calculate vacation days fraction

    if (lifestyle_input_object.get('day_vacation_info') is not None) and \
            (len(lifestyle_input_object.get('day_vacation_info').shape) == 2):
        vacation_perc = lifestyle_input_object.get('day_vacation_info')[:, 1].sum() / \
                        np.shape(lifestyle_input_object.get('day_vacation_info'))[0]

    # get required configs from input data

    NAN_FLOAT_VAL = lifestyle_input_object.get('raw_data_config').get('NAN_FLOAT_VAL')

    logger.debug("%s Using %s as placeholder for Nan values", log_prefix('Generic'), str(NAN_FLOAT_VAL))

    seasons = lifestyle_input_object.get('weather_config').get('season')

    season_fractions = list(map(float, annual_output.get('season_fraction')))

    season_exists = [(sf > 0) for sf in season_fractions]

    day_input_idx = lifestyle_input_object.get('day_input_data_index')

    min_day_idx = int(np.min(day_input_idx))

    max_day_idx = int(np.max(day_input_idx))

    # structure non-temporal payloads before looping over bill cycles

    # ----------------------------ID 1: Seasonal Load Type---------------------------- #

    # Get load type and consumption levels

    seasonal_load_type = annual_output.get('yearly_load_type')

    fn_seasonal_consumption_level = lambda season_name: season_output.get(season_name).get('consumption_level')

    # Get month-season mapping

    monthly_seasons = lifestyle_input_object.get('monthly_seasons')

    MONTH_IDX = lifestyle_input_object.get("raw_data_config").get('MONTH_IDX')
    SEASON_IDX = lifestyle_input_object.get("raw_data_config").get('SEASON_IDX')

    fn_seasonal_months = \
        lambda season_name: list(map(int, monthly_seasons[monthly_seasons[:, SEASON_IDX] == seasons[season_name].value,
                                                          MONTH_IDX])) if season_exists[seasons[season_name].value] else None

    # ----------------------------ID 2: Office Goer ---------------------------- #

    is_office_goer = bool(annual_output.get('is_office_goer'))

    office_goer_prob = float(annual_output.get('office_goer_prob'))

    fn_office_attrs = lambda season_id, attribute: \
        float(season_output.get(seasons(season_id).name).get('office_goer_debug', {}).get(attribute, NAN_FLOAT_VAL)) \
            if season_exists[season_id] else None

    lowcooling_constant = float(lifestyle_output_object.get('lowcooling_constant', NAN_FLOAT_VAL))

    # ----------------------------ID 3, 4: Active/Dormant User ---------------------------- #

    is_active_user = annual_output.get('is_active_user', None)

    active_user_prob = float(annual_output.get('active_user_prob', NAN_FLOAT_VAL))

    is_dormant_user = not is_active_user if (is_active_user is not None) else None

    dormant_user_prob = (1 - active_user_prob) if (active_user_prob < 1) else float(NAN_FLOAT_VAL)

    activity_threshold = float(lifestyle_input_object.get('active_user_config').get('activity_threshold'))

    fn_active_attrs = \
        lambda season_id, attribute: \
            float(season_output.get(seasons(season_id).name).get('active_user_debug', {}).get(attribute, NAN_FLOAT_VAL)) \
                if season_exists[season_id] else None

    fn_active_features =\
        lambda season_id, attribute: \
            list(map(float, season_output.get(seasons(season_id).name).get('active_user_debug', {}).get(attribute, [NAN_FLOAT_VAL])))\
                if season_exists[season_id] else None

    # ----------------------------ID 5: Weekend Warrior ---------------------------- #

    is_weekend_warrior = bool(annual_output.get('is_weekend_warrior'))

    weekend_warrior_prob = float(annual_output.get('weekend_warrior_prob', NAN_FLOAT_VAL))

    fn_weekend_attrs = lambda season_id, attribute: float(season_output.get(
        seasons(season_id).name).get(attribute, NAN_FLOAT_VAL)) if season_exists[season_id] else None

    # ----------------------------ID 6: Hour Fractions(Can get directly) ---------------------------- #

    # ------------------------------------ID 7: Daily Load Type ------------------------------------ #

    fn_billcycle_list_attributes = lambda billcycle_object, attribute: list(map(float, billcycle_object.get(attribute)))\
        if (billcycle_object.get(attribute) is not None) else None

    fn_peaks_info = lambda billcycle_object: billcycle_object.get('peaks') if \
        (billcycle_object.get('peaks') is not None) else None

    # ------------------------------------ID 8: Wakeup Time ------------------------------------ #

    fn_wakeup_attr = lambda billcycle_object, attribute:\
        float(billcycle_object.get('wakeup').get(attribute, NAN_FLOAT_VAL)) if (billcycle_object.get('wakeup') is not None) else None

    # ------------------------------------ID 9: Sleep Time ------------------------------------ #

    fn_sleep_attr = lambda billcycle_object, attribute: \
        float(billcycle_object.get('sleep').get(attribute, NAN_FLOAT_VAL)) if (billcycle_object.get('sleep') is not None) else None

    # Get Billcycle level attributes based on out bill cycles

    out_bill_cycles = lifestyle_input_object.get('out_bill_cycles')

    BILLCYCLE_START_COL = 0

    # remove any out bill cycles which are older than first bill cycle in lifestyle input data

    input_data = lifestyle_input_object.get('input_data')

    min_bill_cycle_val = np.min(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX])

    out_bill_cycles = out_bill_cycles[out_bill_cycles[:, BILLCYCLE_START_COL] >= min_bill_cycle_val]

    # Create Final Non Temporal ids profile output

    non_temporal_profile = {

        "lifestyleid_1": {
            "name"           : "SeasonalLoadType",
            "value"          : seasonal_load_type,
            "attributes"     : {
                "consumptionLevelWinter"    : fn_seasonal_consumption_level('winter'),
                "consumptionLevelSummer"    : fn_seasonal_consumption_level('summer'),
                "consumptionLevelTransition": fn_seasonal_consumption_level('transition'),
            },
            "debugAttributes": {
                "winterMonths"    : fn_seasonal_months('winter'),
                "summerMonths"    : fn_seasonal_months('summer'),
                "transitionMonths": fn_seasonal_months('transition'),

            }
        },
        "lifestyleid_2": {
            "name"           : "OfficeGoer",
            "value"          : is_office_goer,
            "attributes"     : {
                "officeGoerProbability": office_goer_prob
            },
            "debugAttributes": {
                "lowcoolingConstant"     : lowcooling_constant,
                "seasonKeys"             : [seasons(idx).name for idx in range(seasons.count.value)],
                "seasonalProbabilities"  : [fn_office_attrs(idx, 'office_goer_primary_prob') for idx in
                                            range(seasons.count.value)],
                "lowcoolingProbabilities": [fn_office_attrs(idx, 'office_lowcooling_prob') for idx in
                                            range(seasons.count.value)]
            }
        },
        "lifestyleid_3": {
            "name"           : "ActiveUser",
            "value"          : is_active_user,
            "attributes"     : {
                "activeUserProbability": active_user_prob
            },
            "debugAttributes": {
                "seasonKeys"              : [seasons(idx).name for idx in range(seasons.count.value)],
                "seasonFractions"         : list(season_fractions),
                "seasonalProbabilities"   : [fn_active_attrs(idx, 'active_user_prob') for idx in
                                             range(seasons.count.value)],
                "baseloadConsumption"     : [fn_active_attrs(idx, 'baseload') for idx in
                                             range(seasons.count.value)],
                "totalConsumption"        : [fn_active_attrs(idx, 'total_consumption') for idx in
                                             range(seasons.count.value)],
                "nonBaseloadFraction"     : [fn_active_attrs(idx, 'non_baseload_fraction') for idx in
                                             range(seasons.count.value)],
                "averageActivity"         : [fn_active_attrs(idx, 'average_activity') for idx in
                                             range(seasons.count.value)],
                "withinDayDeviation"      : [fn_active_attrs(idx, 'within_day_deviation') for idx in
                                             range(seasons.count.value)],
                "acrossDayDeviation"      : [fn_active_attrs(idx, 'across_day_deviation') for idx in
                                             range(seasons.count.value)],
                "winterFeaturesNormed"    : fn_active_features(seasons['winter'].value, 'features_normed'),
                "summerFeaturesNormed"    : fn_active_features(seasons['summer'].value, 'features_normed'),
                "transitionFeaturesNormed": fn_active_features(seasons['transition'].value, 'features_normed'),
                "activityThreshold"       : activity_threshold
            }
        },
        "lifestyleid_4": {
            "name"           : "DormantUser",
            "value"          : is_dormant_user,
            "attributes"     : {
                "dormantUserProbability": dormant_user_prob
            },
            "debugAttributes": {

            }
        },
        "lifestyleid_5": {
            "name"           : "WeekendWarrior",
            "value"          : is_weekend_warrior,
            "attributes"     : {
                "weekendWarriorProbability": weekend_warrior_prob
            },
            "debugAttributes": {
                "seasonKeys"           : [seasons(idx).name for idx in range(seasons.count.value)],
                "seasonalProbabilities": [fn_weekend_attrs(idx, 'weekend_warrior_prob') for idx in
                                          range(seasons.count.value)],
            }
        },
    }

    for billcycle_start, billcycle_end in out_bill_cycles:
        # Extract billcycle level output

        billcycle_output = lifestyle_output_object.get('billcycle').get(billcycle_start)

        # fill temporal profile

        billcycle_profile = {
            "start"         : int(billcycle_start),
            "end"           : int(billcycle_end),
            "dataRange"     : {
                "start": min_day_idx,
                "end"  : max_day_idx
            },
            "validity"      : {
                "start": min_day_idx,
                "end"  : max_day_idx
            },
            "lifestyleid_6" : {
                "name"           : "HourFractions",
                "value"          : fn_billcycle_list_attributes(billcycle_output, 'hour_fraction'),
                "attributes"     : {
                    "hourFractionWeekday": fn_billcycle_list_attributes(billcycle_output, 'weekday_hour_fraction'),
                    "hourFractionWeekend": fn_billcycle_list_attributes(billcycle_output, 'weekend_hour_fraction')
                },
                "debugAttributes": {

                }
            },
            "lifestyleid_7" : {
                "name"           : "DailyLoadType",
                "value"          : dominant_load_type,
                "attributes"     : {
                    "loadtypeConfidence"     : dominant_load_fraction,
                    "consumptionLevel"       : annual_output.get('consumption_level', 'not_known'),
                    "clusterNames"           : [daily_load_type(idx).name for idx in
                                                range(daily_load_type.count.value)],
                    "clusterFractionsAll": fn_billcycle_list_attributes(event_output, 'cluster_fraction'),
                    "clusterFractionsWeekday": fn_billcycle_list_attributes(event_output, 'weekday_cluster_fraction'),
                    "clusterFractionsWeekend": fn_billcycle_list_attributes(event_output, 'weekend_cluster_fraction')
                },
                "debugAttributes": {
                    "peaksInfo": str(fn_peaks_info(billcycle_output))
                }
            },
            "lifestyleid_8" : {
                "name"           : "WakeUpTime",
                "value"          : fn_wakeup_attr(billcycle_output, 'wakeup_time'),
                "attributes"     : {
                    "confidenceInterval": fn_wakeup_attr(billcycle_output, 'wakeup_confidence')
                },
                "debugAttributes": {

                }

            },
            "lifestyleid_9" : {
                "name"           : "SleepTime",
                "value"          : fn_sleep_attr(billcycle_output, 'sleep_time'),
                "attributes"     : {
                    "confidenceInterval": fn_sleep_attr(billcycle_output, 'sleep_confidence')
                },
                "debugAttributes": {

                }
            },
            "lifestyleid_10": {
                "name"           : "VacationPercentage",
                "value"          : vacation_perc,
                "attributes"     : {

                },
                "debugAttributes": {

                }
            }

        }

        # Update with non temporal profile

        billcycle_profile.update(non_temporal_profile)

        # update billcycle profile in disagg_output_object

        disagg_output_object['lifestyle_profile'][int(billcycle_start)]['profileList'][0].update(billcycle_profile)

        # lifestyle billcycle profile validation

        validate_lifestyle_profile_schema_for_billcycle(disagg_output_object, billcycle_start, logger_pass)

    t_populate_lifestyle_profile_end = datetime.now()

    logger.info("%s Populated lifestyle profile in | %.3f s", log_prefix('Generic'),
                get_time_diff(t_populate_lifestyle_profile_start,
                              t_populate_lifestyle_profile_end))

    return disagg_output_object
