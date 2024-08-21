"""
Author - Prasoon Patidar
Date - 3rd June 2020
Contains static config for lifestyle modules
"""

# import python packages

from enum import Enum


class DailyLoadType(Enum):

    """
    Enum for Various Daily Load Types
    """

    UNDEFINED = 0
    BASE_LOAD = 1
    DAY_LOAD = 2
    NIGHT_LOAD = 3
    EARLY_NOON = 4
    LATE_NOON = 5
    EARLY_EVE = 6
    LATE_EVE = 7
    NIGHT_EVE = 8
    MORNING_EVE = 9
    NOON_EVE = 10
    EARLY_MORN = 11
    LATE_MORN = 12
    NOON_LOAD = 13
    NIGHT_NOON = 14
    EVENING_LOAD = 15
    count = 16


class YearlyLoadType(Enum):

    """
    Enum for Various Daily Load Types
    """

    UNDEFINED = 0
    ALL_YEAR_PEAK = 1
    ALL_YEAR_BASELOAD = 2
    MILD_SUMMER_PEAK = 3
    HEAVY_SUMMER_PEAK = 4
    MILD_WINTER_PEAK = 5
    HEAVY_WINTER_PEAK = 6
    HEAVY_SUMMER_HEAVY_WINTER = 7
    MILD_SUMMER_MILD_WINTER = 8
    HEAVY_SUMMER_MILD_WINTER = 9
    MILD_SUMMER_HEAVY_WINTER = 10
    count = 11


class DwellingType(Enum):

    """
    Enum for dwelling types-id mapping from meta data
    """

    NOT_KNOWN = 0
    SINGLE = 1
    APARTMENTS = 2
    CONDO = 3
    MULTIPLEX = 4
    DETACHED = 5
    BUNGALOW = 6
    count = 7


class Season(Enum):

    """
    Enum for various all the seasons
    """

    winter = 0
    summer = 1
    transition = 2
    count = 3


class ConsumptionLevel(Enum):

    """
    Enum for consumption levels
    """

    not_known = 0
    low = 1
    low_mid = 2
    mid = 3
    mid_high = 4
    high = 5
    count = 6


class LifestyleId(Enum):

    """
    Enum for lifestyle ids in final profile
    """

    Generic = 0
    SeasonalLoadType = 1
    OfficeGoer = 2
    ActiveUser = 3
    DormantUser = 4
    WeekendWarrior = 5
    HourFractions = 6
    DailyLoadType = 7
    WakeUpTime = 8
    SleepTime = 9
    VacationPercentage = 10
    count = 11


def init_lifestyle_config():

    """
    Returns:
        lifestyle_config(dict)                     : Dictionary containing static config for lifestyle

    """

    lifestyle_config = {

        # Sub-Config for preprocessing Input Data

        'raw_data_config'       : {
            'num_days_limit'    : 365,
            'max_percentile_val': 97,
            'SUNDAY_DAY_ID'     : 1,
            'SATURDAY_DAY_ID'   : 7,
            'NAN_FLOAT_VAL'     : -14001014e-8,
            'MONTH_IDX'         : 0,
            'TEMP_IDX'          : 1,
            'HDD_IDX'           : 2,
            'CDD_IDX'           : 3,
            'SEASON_IDX'        : 4,
            "NEW_IDX_COUNT"     : 2
        },

        # 'num_days_limit': Maximum number of days to run lifestyle module
        # 'max_percentile_val': Percentile value to filter input data
        # 'SUNDAY_DAY_ID': sunday id
        # 'SATURDAY_DAY_ID': saturday id
        # 'NAN_FLOAT_VAL': default value for nan
        # 'MONTH_IDX': month index
        # 'SEASON_IDX': season index

        # Sub Config for user kmeans model

        'kmeans_model_config'   : {
            # allowed (median,iqd) bucket for processing user
            'allowed_user_buckets': [(1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)],
            'model_id_map'        : ['lcav', 'lchv', 'aclv', 'acav', 'achv', 'hcav', 'hchv'],
            'default_user_bucket' : (2, 2)

        },

        # 'allowed_user_buckets': Allowed set of consumption buckets
        # 'model_id_map': mapping of each allowed buckets
        # lc - low consumption
        # ac - average consumption
        # hc - high consumption
        # lv - low variation
        # av - mid variation
        # hv - high variation
        # 'default_user_bucket': Defult user bucket

        # Sub-Config for Weather Info Module

        'weather_config'        : {

            # Enum for seasons

            'season'               : Season,

            # Seasonal CDD HDD Limits

            's2t_cdd'              : 100,
            's2t_hdd'              : 75,

            't2s_cdd_hdd_diff'     : 100,
            't2s_hdd'              : 50,

            'transition_month_perc': 0.25,
            'temperature_setpoint' : 65,

            'default_setpoint': 65,
        },

        # 'season': Season,
        # 's2t_cdd': season to transition cdd limit
        # 's2t_hdd': season to transition hdd limit
        # 't2s_cdd_hdd_diff': transition to season cdd hdd difference
        # 't2s_hdd': transition to season hdd
        # 'transition_month_perc': percentage of transition month
        # 'temperature_setpoint': setpoint

        # Enum for Daily Load Types

        'daily_load_type'       : DailyLoadType,

        # Enum for Yearly Load Types

        'yearly_load_type'      : YearlyLoadType,

        # Enum for various Dwelling Types

        'dwelling_type'         : DwellingType,

        # Enum for different consumption levels

        'consumption_level'     : ConsumptionLevel,

        # Enum for lifestyle ids for logging purposes

        'lifestyle_ids'         : LifestyleId,

        'yearly_load_type_list': ['UNDEFINED', 'ALL_YEAR_PEAK', 'ALL_YEAR_BASELOAD', 'MILD_SUMMER_PEAK', 'HEAVY_SUMMER_PEAK',
                                  'MILD_WINTER_PEAK', 'HEAVY_WINTER_PEAK', 'HEAVY_SUMMER_HEAVY_WINTER', 'MILD_SUMMER_MILD_WINTER',
                                  'HEAVY_SUMMER_MILD_WINTER', 'MILD_SUMMER_HEAVY_WINTER'],

        # Sub-Config for Wakeup/Sleep Detection Module

        'wakeup_sleep_config'   : {

            'confidence_band_length_ratio': 0.25,
            'sunrise_start_hour_limit'    : 0,
            'sunrise_end_hour_limit'      : 12,
            'sunset_start_hour_limit'     : 12,
            'sunset_end_hour_limit'       : 23,
            'allowed_sunrise_deviation'   : 1.5,
            'wakeup_band_start_hour_min'  : 3,
            'allowed_vacation_percentage' : 70

        },

        # 'confidence_band_length_ratio': wakeup band confidence ratio
        # 'sunrise_start_hour_limit': minimum sunrise hour
        # 'sunrise_end_hour_limit': maximum sunrise hour
        # 'sunset_start_hour_limit': minimum sunset hour
        # 'sunset_end_hour_limit': maximum sunset hour
        # 'allowed_sunrise_deviation': maximum deviation in sunrise
        # 'wakeup_band_start_hour_min': minimum wakeup time ,
        # 'allowed_vacation_percentage': Maximum vacation percentage


        # Sub-Config for Peak Detection Module

        'peak_detection_config' : {

            # allowed cluster fraction limit for detecting peaks
            'allowed_cluster_fraction_limit': 0.7,

            # Indices for various peak detection parameters for various peak types
            'LAG_IDX'                       : 0,
            'THRESHOLD_IDX'                 : 1,
            'INFLUENCE_IDX'                 : 2,

            # Config for Various Peak Types
            'REFERENCE_PEAKS'               : [3, 5, 0.05],
            'CONSUMPTION_PEAKS'             : [5, 3, 0.2],

            # deviation ratio wrt peak duration
            'peak_duration_deviation_ratio' : 0.75,
            'peak_duration_deviation_min'   : 2
        },

        # 'allowed_cluster_fraction_limit': Minimum allowed cluster fraction limit
        # 'LAG_IDX': lag duration, no. of lag hours to consider for peak identification
        # 'THRESHOLD_IDX': threshold to classify as peak/nonpeak shift
        # 'INFLUENCE_IDX': influence of new hour on moving mean and std values

        # Sub-config for lowcooling constant caluclation

        'lowcooling_config'     : {

            # configs to get lowcooling constant

            'cdd_bincount'                              : 8,
            'bucket_normed_cons_percentile'             : 25,
            'start_hour'                                : 9,
            'end_hour'                                  : 16,
            'weekday_lowcooling_fraction_agg_percentile': 50,

        },

        # 'cdd_bincount': Number of cdd buckets,
        # 'bucket_normed_cons_percentile': Base cooling percentile ,
        # 'start_hour' - 'end_hour': Allowed low cooling hours

        # Sub-Config for Office Goer Module

        'office_goer_config'    : {
            'OFFICE_CLUSTERS_PRIMARY'                  : [DailyLoadType.NIGHT_LOAD,
                                                          DailyLoadType.EARLY_EVE,
                                                          DailyLoadType.LATE_EVE,
                                                          DailyLoadType.NIGHT_EVE,
                                                          DailyLoadType.MORNING_EVE,
                                                          DailyLoadType.BASE_LOAD],
            'OFFICE_CLUSTERS_NONWINTER'                : [DailyLoadType.NOON_EVE,
                                                          DailyLoadType.LATE_NOON],
            'MIN_NONPEAK_CONSUMPTION_RATIO'            : 0.2,
            'MIN_LOW_COOLING_RATIO'                    : 0.1,
            'MIN_DAYS_CLUSTER_FRACTION'                : 0.05,
            'NONPEAK_CONSUMPTION_PERCENTILE'           : 50,

            # get break threshold for tp peak detection

            'peak_detection_break_threshold_percentile': 20,

            # morning and evening peak info config

            'morning_peak_start_min_hour'              : 5,
            'morning_peak_end_max_hour'                : 12,
            'evening_peak_start_min_hour'              : 15,
            'best_morning_peak_start_limit'            : 5,
            'best_morning_peak_end_limit'              : 10,
            'best_evening_peak_start_limit'            : 16,
            'best_evening_peak_end_limit'              : 22,

            # distance/duration scoring and filtering

            'duration_threshold_in_hour'               : 3,
            'distance_threshold_in_hour'               : 4,
            'duration_scoring_weight'                  : 0.5,
            'distance_scoring_weight'                  : 1,

            # office goer annual module config

            'office_goer_prob_threshold'               : 0.6,
            'summer_low_prob_threshold'                : 0.4,
            'winter_low_prob_threshold'                : 0.6,
            'winter_high_prob_threshold'               : 0.7,
            'transition_high_prob_threshold'           : 0.7,

            'office_goer_score_soft_margin'            : 0.04
        },

        # 'OFFICE_CLUSTERS_PRIMARY': Office goer cluster types
        # 'OFFICE_CLUSTERS_NONWINTER': Low cooling probability cluster types

        # 'morning_peak_start_min_hour' - 'morning_peak_end_max_hour': allowed morning peak hours
        # 'evening_peak_start_min_hour':  allowed minimum evening peak hours
        # 'best_morning_peak_start_limit' - 'best_morning_peak_end_limit': Preferred morning peak hours
        # 'best_evening_peak_start_limit' - 'best_evening_peak_end_limit': Preferred evening peak hours

        # distance_threshold : distance limit between peaks birth time to qualify as close peak
        # duration_threshold : difference in peak duration limit to qualify as a close peak
        # distance_weight : weight assigned to difference in distance for scoring
        # duration_weight : weight assigned to difference in duration for scoring

        # Sub-Config for Weekend Warrior Module

        'weekend_warrior_config': {
            'MAX_PEAK_NORMED_AREA_FRACTION'   : 0.7,
            'MAX_CONSUMPTION_DIFFERENCE_RATIO': 0.5,
            'MIN_SEASON_FRACTION'             : 0.4,

            'WEEKEND_WARRIOR_PEAKS'           : [3, 0.5, 0.1],

            'day_start_hour'                  : 6,
            'day_end_hour'                    : 24,

            'weekend_warrior_prob_threshold'  : 0.6,

            'weekend_warrior_soft_margin'     : 0.04
        },

        # 'day_start_hour' - 'day_end_hour': Hours used for calculation

        # Sub-Config for Active User Module

        'active_user_config'    : {

            # attribute level configs

            'baseload_percentile'                      : 1,
            'within_day_low_percentile'                : 10,
            'within_day_high_percentile'               : 90,
            'activity_threshold'                       : 200,
            'activity_start_hour'                      : 4,
            'feature_count'                            : 4,

            # indices for various user attributes in Active User Configs

            'NON_BASELOAD_FRACTION_IDX'                : 0,
            'WITHIN_DAY_DEVIATION_IDX'                 : 1,
            'ACROSS_DAY_DEVIATION_IDX'                 : 2,
            'AVERAGE_ACTIVITY_IDX'                     : 3,

            # indices for different values for each user attribute

            'MIN_VAL_IDX'                              : 0,
            'MAX_VAL_IDX'                              : 1,
            'WEIGHT_IDX'                               : 2,

            # Seasonal config weights for Active User Module
            # list idx corresponds value set for user attributes
            # list item idx corresponds to value for each attributes

            'transition'                               : [(0, 1, 1),
                                                          (0, 2465, 1),
                                                          (0, 45000, 0),
                                                          (0, 8.4, 4)],
            'winter'                                   : [(0, 1, 1),
                                                          (0, 2600, 0.5),
                                                          (0, 57700, 1),
                                                          (0, 8.3, 3)],
            'summer'                                   : [(0, 1, 0),
                                                          (0, 2500, 0.5),
                                                          (0, 46300, 0.5),
                                                          (0, 8.4, 2)],

            # active user annual prob config

            'active_user_prob_threshold'               : 0.5,
            'transition_season_fraction_high_threshold': 0.5,
            'summer_season_fraction_low_threshold'     : 0.3,
            'summer_season_fraction_high_threshold'    : 0.6,
            'summer_season_fraction_static'            : 0.3,

            'active_user_prob_soft_margin'             : 0.04
        },

        # debug config for lifestyle

        'debug_config'          : {

            # set debug mode to true or false

            'debug_mode'     : False,

            # directories to dump plots and results

            'plot_dir'       : '/var/log/bidgely/lifestyle/debug_plots',
            'result_dump_dir': '/var/log/bidgely/lifestyle/result_dump'
        }

    }

    return lifestyle_config
