"""
Author - Sahana M
Date - 28/10/2021
Contains all the configurations required for weather data extraction and integration
"""


class Weatherconfigs:
    """Constants to use in Weather API Integration"""

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
    # Epoch timestamp representing sunrise
    INPUT_SUNRISE_IDX = 7
    # Epoch timestamp representing sunset
    INPUT_SUNSET_IDX = 8
    # Temperature in Fahrenheit
    INPUT_TEMPERATURE_IDX = 9
    # Cloud Cover as percentage
    INPUT_SKYCOV_IDX = 10
    # Wind speed in miles per hour
    INPUT_WIND_SPD_IDX = 11
    # Dew point temperature in Fahrenheit
    INPUT_DEW_IDX = 12
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
    # Water heater potential
    INPUT_WH_POTENTIAL_IDX = 24
    # Is cold event
    INPUT_COLD_EVENT_IDX = 25
    # Is hot event
    INPUT_HOT_EVENT_IDX = 26
    # Season Label (-1 = Winter, -0.5 = Transition winter, 0 = Transition, 0.5 = Transition summer, 1 = Summer)
    INPUT_S_LABEL_IDX = 27

    DROP_COLUMNS = ['dayStartTimestamp', 'lastUpdatedTimestamp', 'timestamp']

    INPUT_COLS_ORDER = ['temperature', 'cloudCeiling', 'windSpeed', 'dewPoint', 'feelsLike', 'precipitation', 'snow',
                        'seaLevelPressure', 'spcHumidity', 'relHumidity', 'wetBulb', 'windDirection', 'visibility',
                        'coolingPot', 'heatingPot', 'whPot', 'isColdEvent', 'isHotEvent', 'sLabel', 'dayStartTimestamp',
                        'lastUpdatedTimestamp', 'timestamp']

    FINAL_COLS_ORDER = ['temperature', 'cloudCeiling', 'windSpeed', 'dewPoint', 'feelsLike', 'precipitation', 'snow',
                        'seaLevelPressure', 'spcHumidity', 'relHumidity', 'wetBulb', 'windDirection', 'visibility',
                        'coolingPot', 'heatingPot', 'whPot', 'isColdEvent', 'isHotEvent', 'sLabel']
