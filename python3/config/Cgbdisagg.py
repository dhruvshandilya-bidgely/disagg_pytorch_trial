"""Input data format"""


class Cgbdisagg:
    """Constants to use in GB Disagg"""

    # Indices for the 28 column data matrix

    INPUT_DIMENSION = 28
    # Unix time pointing to start of the billing cycle
    INPUT_BILL_CYCLE_IDX = 0
    # Unix time pointing to start of the week
    INPUT_WEEK_IDX = 1
    # Unix time pointing to start of the day
    INPUT_DAY_IDX = 2
    # Day of week - 1:7, 1 = sunday, 7 = saturday
    INPUT_DOW_IDX = 3
    # Hour of day - 0:23
    INPUT_HOD_IDX = 4
    # Unix time pointing to start of each epoch.
    INPUT_EPOCH_IDX = 5
    # Energy Consumption in the Epoch [Watt-hour]
    INPUT_CONSUMPTION_IDX = 6
    # Temperature in Fahrenheit
    INPUT_TEMPERATURE_IDX = 7
    # Cloud Cover as percentage
    INPUT_SKYCOV_IDX = 8
    # Wind speed in miles per hour
    INPUT_WIND_SPD_IDX = 9
    # Dew point temperature in Fahrenheit
    INPUT_DEW_IDX = 10
    # Epoch timestamp representing sunrise
    INPUT_SUNRISE_IDX = 11
    # Epoch timestamp representing sunset
    INPUT_SUNSET_IDX = 12
    # Feels Like Temperature in Fahrenheit
    INPUT_FEELS_LIKE_IDX = 13
    # Precipitation in inches
    INPUT_PREC_IDX = 14
    # Snowfall in inches
    INPUT_SNOW_IDX = 15
    # Sea level pressure in milibars
    INPUT_SL_PRESS_IDX = 16
    # Specific humidity in g / kg
    INPUT_SPC_HUM_IDX = 17
    # Relative humidity in %
    INPUT_REL_HUM_IDX = 18
    # Wet bulb temperature in Fahrenheit
    INPUT_WET_BULB_IDX = 19
    # Wind direction East = 90, 180 = South, West = 270, 360 = North, 0 = No wind
    INPUT_WIND_DIR_IDX = 20
    # Visibility
    INPUT_VISIBILITY_IDX = 21
    # Cooling potential
    INPUT_COOLING_POTENTIAL_IDX = 22
    # Heating potential
    INPUT_HEATING_POTENTIAL_IDX = 23
    # Water Heater potential
    INPUT_WH_POTENTIAL_IDX = 24
    # Is cold event
    INPUT_COLD_EVENT_IDX = 25
    # Is hot event
    INPUT_HOT_EVENT_IDX = 26
    # Season Label (-1 = Winter, -0.5 = Transition winter, 0 = Transition, 0.5 = Transition summer, 1 = Summer)
    INPUT_S_LABEL_IDX = 27

    # Column names for 21 column matrix

    INPUT_COLUMN_NAMES = ['month', 'week', 'day', 'day_of_week', 'hour_of_day', 'epoch', 'consumption', 'temperature',
                          'sky_cover', 'wind', 'dew', 'sunrise', 'sunset', 'feels_like', 'precipitation', 'snow',
                          'pressure', 'sp_humid', 'rel_hum', 'wet_bulb', 'wind_dir', 'visibility', 'cooling_pot',
                          'heating_pot', 'wh_pot', 'is_cold_event', 'is_hot_event', 's_label']

    # Few constants useful in calculating period related values

    SEC_IN_WEEK = 7 * 86400
    SEC_IN_DAY = 86400
    SEC_IN_HOUR = 3600
    SEC_IN_30_MIN = 1800
    SEC_IN_15_MIN = 900
    SEC_IN_1_MIN = 60
    HRS_IN_DAY = 24
    DAYS_IN_WEEK = 7
    DAYS_IN_MONTH = 30
    MONTHS_IN_YEAR = 12
    DAYS_IN_YEAR = 365
    WH_IN_1_KWH = 1000

    # Constants we need for the run to be configured

    QUEUE_RETRY_TIME = 2
    QUEUE_BASE_NAME = "GbTempDataReadyEventPyAmi"
    QUEUE_BASE_NAME_PRIORITY = "GbTempDataReadyEventPyAmiPriority"

    PROCESSES_MULTIPLIER_PRIORITY_DEFAULT = 1.75
    PROCESSES_MULTIPLIER_DEFAULT = 1.75

    API_RETRY_TIME = 2
    MAX_TRIES_PRIMARY_API = 2
    MAX_TRIES_SECONDARY_API = 2
    MAX_TRIES_PRIMARY_API_PRIORITY = 1

    HIGH_CONS_THRESHOLD = 20000

    APPS_TO_PULL_HSM_FOR = ['pp', 'wh', 'ao', 'ref', 'hvac', 'solar', 'ev', 'li', 'va']

    # Constants needed for pre and post disagg ops

    MIN_TEMP_NOT_INTERPOLATE = 90

    # Weather data cache constants
    ENABLE_WEATHER_DATA_CACHE = True

    # 10 GB Max size limit
    ON_DISK_WEATHER_DATA_CACHE_SIZE_LIMIT = int(16e9)

    # Cache expiry time
    # None means no expiry, use number of seconds to set expire time
    WEATHER_DATA_CACHE_EXPIRE_TIME = None

    # Default client id and secret for oauth token fetch

    CLIENT_ID_DICT = {
        'dev': 'admin',
        'ds': 'admin',
        'nonprodqa': 'admin',
        'prod-na': '',
        'prod-eu': '',
        'prod-jp': '',
        'prod-ca': '',
        'prod-na-2': '',
        'preprod-na': 'admin',
        'qaperfenv': 'admin',
        'uat': 'admin',
        'productqa': ''
    }

    CLIENT_SECRET_DICT = {
        'dev': 'admin',
        'ds': 'admin',
        'nonprodqa': 'admin',
        'prod-na': '',
        'prod-eu': '',
        'prod-jp': '',
        'prod-ca': '',
        'prod-na-2': '',
        'preprod-na': 'admin',
        'qaperfenv': 'admin',
        'uat': 'admin',
        'productqa': ''
    }
