"""
Author - Paras Tehria
Date - 17-Nov-2020
This module is used to get epoch level irradiance
"""

import pysolar
import datetime
import numpy as np
from pytz import timezone


def get_irradiance(epoch_time, latitude, longitude, user_timezone):
    """
        Get Irradiance for all epochs timestamps

        Parameters:
            epoch_time              (np.ndarray)        : numpy array containing epoch_timestamp
            latitude                (float)              : user's latitude
            longitude               (float)              : user's longitude
            user_timezone           (dict)              : user's timezone

        Returns:
            irradiance_arr          (np.ndarray)        : numpy array containing irradiance for each input timestamp
    """

    date = [datetime.datetime.fromtimestamp(x, timezone(user_timezone)) for x in epoch_time]
    date = np.array(date)

    # Getting altitude degree using pysolar package
    altitude_deg = pysolar.solar.get_altitude_fast(latitude, longitude, date)

    date_altitude = np.c_[date, altitude_deg]
    epoch_time = epoch_time[~np.isnan(date_altitude[:, 1].astype(float))]
    date_altitude = date_altitude[~np.isnan(date_altitude[:, 1].astype(float))]

    # Looping over each timestamp to calculate irradiance
    irradiance = [pysolar.solar.radiation.get_radiation_direct(x[0], x[1]) for x in date_altitude]

    irradiance_arr = np.c_[epoch_time, irradiance]
    return irradiance_arr
