"""
Author - Mayank Sharan
Date - 21st May 2021
Constants used to enforce sanity constants
"""

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


class SanityConstants:

    """
    Class containing constants for sanity minimum and maximum
    """

    sanity_min_limits = {
        Cgbdisagg.INPUT_BILL_CYCLE_IDX: 0,
        Cgbdisagg.INPUT_WEEK_IDX: 0,
        Cgbdisagg.INPUT_DAY_IDX: 0,
        Cgbdisagg.INPUT_DOW_IDX: 1,
        Cgbdisagg.INPUT_HOD_IDX: 0,
        Cgbdisagg.INPUT_EPOCH_IDX: 0,
        Cgbdisagg.INPUT_CONSUMPTION_IDX: 0,
        Cgbdisagg.INPUT_TEMPERATURE_IDX: -148,
        Cgbdisagg.INPUT_SKYCOV_IDX: 0,
        Cgbdisagg.INPUT_WIND_SPD_IDX: 0,
        Cgbdisagg.INPUT_DEW_IDX: -148,
        Cgbdisagg.INPUT_SUNRISE_IDX: 0,
        Cgbdisagg.INPUT_SUNSET_IDX: 0,
        Cgbdisagg.INPUT_FEELS_LIKE_IDX: -148,
        Cgbdisagg.INPUT_PREC_IDX: 0,
        Cgbdisagg.INPUT_SNOW_IDX: 0,
        Cgbdisagg.INPUT_SL_PRESS_IDX: 0,
        Cgbdisagg.INPUT_SPC_HUM_IDX: 0,
        Cgbdisagg.INPUT_REL_HUM_IDX: 0,
        Cgbdisagg.INPUT_WET_BULB_IDX: -148,
        Cgbdisagg.INPUT_WIND_DIR_IDX: 0,
    }

    sanity_max_limits = {
        Cgbdisagg.INPUT_BILL_CYCLE_IDX: 3000000000,
        Cgbdisagg.INPUT_WEEK_IDX: 3000000000,
        Cgbdisagg.INPUT_DAY_IDX: 3000000000,
        Cgbdisagg.INPUT_DOW_IDX: 7,
        Cgbdisagg.INPUT_HOD_IDX: 23,
        Cgbdisagg.INPUT_EPOCH_IDX: 3000000000,
        Cgbdisagg.INPUT_CONSUMPTION_IDX: 10000000,
        Cgbdisagg.INPUT_TEMPERATURE_IDX: 158,
        Cgbdisagg.INPUT_SKYCOV_IDX: 100,
        Cgbdisagg.INPUT_WIND_SPD_IDX: 500,
        Cgbdisagg.INPUT_DEW_IDX: 158,
        Cgbdisagg.INPUT_SUNRISE_IDX: 3000000000,
        Cgbdisagg.INPUT_SUNSET_IDX: 3000000000,
        Cgbdisagg.INPUT_FEELS_LIKE_IDX: 158,
        Cgbdisagg.INPUT_PREC_IDX: 20,
        Cgbdisagg.INPUT_SNOW_IDX: 20,
        Cgbdisagg.INPUT_SL_PRESS_IDX: 1500,
        Cgbdisagg.INPUT_SPC_HUM_IDX: 50,
        Cgbdisagg.INPUT_REL_HUM_IDX: 100,
        Cgbdisagg.INPUT_WET_BULB_IDX: 158,
        Cgbdisagg.INPUT_WIND_DIR_IDX: 360,
    }
