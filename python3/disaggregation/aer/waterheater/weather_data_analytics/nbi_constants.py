"""
Author: Mayank Sharan
Created: 6-Jan-2020
Contains classes defining constants needed in the engine
"""


class TimeConstants:

    """Class for different time constants"""

    # Seconds in 1 minute
    sec_in_1_min = 60

    # Seconds in 1 hour
    sec_in_1_hr = 3600

    # Seconds in 1 day
    sec_in_1_day = 86400

    # Seconds in 1 day
    hr_in_1_day = 24

    # Max days in 1 month
    max_days_in_1_month = 31

    # days in 1 year
    days_in_1_year = 365

    # freshness scaling constant
    freshness_exp_life = 3 * max_days_in_1_month * sec_in_1_day

    # default timestamp for time in future
    future_ts = 1700000000

    # default timestamp for time in past
    past_ts = 0

    # time period for which cached config is valid in seconds
    config_valid_time = 900

    # Time before we retry the SQS queue
    queue_retry_time = 2


class AppConstants:

    """Class for constants related to appliances"""

    # App code to name mapping
    # to    : Total consumption / Whole home
    # pp    : Pool pump
    # sh    : Space heating
    # ac    : Air Conditioning
    # wh    : Water Heating
    # ao    : Always On
    # ref   : Refrigeration
    # ev    : Electric vehicle
    # li    : Lighting
    # va    : Vacation type 1
    # vad   : Vacation type 2 - disconnection

    # Map of appliance name to app id

    app_name_id_map = {
        'pp': 2,
        'sh': 3,
        'ac': 4,
        'wh': 7,
        'ao': 8,
        'ref': 9,
        'to': 17,
        'ev': 18,
        'li': 71,
        'va': 1001,
        'vad': 1002,
    }

    # Map of appliance id to app name

    app_id_name_map = {
        2: 'pp',
        3: 'sh',
        4: 'ac',
        7: 'wh',
        8: 'ao',
        9: 'ref',
        17: 'to',
        18: 'ev',
        71: 'li',
        1001: 'va',
        1002: 'vad',
    }

    # Map of appliance name to scaling factor

    app_scaling_factor = {
        'to': 1,
        'pp': 2.5,
        'sh': 1.2,
        'ac': 1.2,
        'wh': 2,
        'ao': 3,
        'ref': 4,
        'ev': 2,
        'li': 4,
    }

    # List of appliance names

    app_name_arr = ['ac', 'ao', 'ev', 'hvac', 'li', 'pp', 'ref', 'sh', 'to', 'wh', 'ck', 'ld', 'en']

    # Map of full appliance name to app id

    full_app_name_id_map = {
        'poolsandsaunasoutput': 2,
        'spaceheatingoutput': 3,
        'ac_output': 4,
        'waterheatingoutput': 7,
        'alwaysonoutput': 8,
        'refrigerationoutput': 9,
        'totaloutput': 17,
        'lightingoutput': 71,
        'otheroutput': 99
    }


class ScoringConstants:

    """Class for constants used to compute scores"""

    # Weightages for combining relevance score of insight and value score of action into an interaction score

    rel_weight = 0.6
    val_weight = 0.4

    # Weightages for combining kwh and pct/percentile into a score

    pct_weight = 0.4
    ptile_weight = 0.4
    kwh_weight = 0.6
    cost_weight = 0.6

    # kwh to $ conversion factor. Currently a naive hard coded 10 cents per kWh

    dollar_per_kwh = 0.1


class FilterConstants:

    """Class for constants used to populate filters"""

    # Populate possible values for ownership filter

    ownership_filters = ['owner', 'renter']

    # Populate possible values for income filter

    income_filters = ['low', 'med', 'high']

    # Populate possible values for season filter

    season_filters = ['summer', 'winter', 'other']

    # Populate appliances to run appliance filter for. Excludes universally assumed or non true disagg appliance

    app_filters = ['ac', 'sh', 'wh', 'ev', 'pp', 'hvac']


class ConfigConstants:

    """Class for constants related to config"""

    default_pilot_id = -1


class ExitCodes:

    """Class for exit code constants"""

    success_exit_code = 0
    engine_run_failure_code = -1
    data_prep_failure_code = -2
