
"""
Author - Nisha Agarwal
Date - 8th Oct 20
Calculate final wake up and sleep time
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_wakeup_sleep_time(samples_per_hour, sleep_hours, config):

    """
    Calculate wakeup and sleep time of the user

    Parameters:
        samples_per_hour         (int)          : number of samples in an hour
        sleep_hours              (np.ndarray)   : boolean array for sleep hours
        config                   (dict)         : config dictionary

    Returns:
        wakeup_time              (float)        : String format of Wakeup time of the user
        sleep_time               (float)        : String format of Sleep time of the user
        wakeup_time_int          (float)        : Numeric format of Wakeup time of the user
        sleep_time_int           (float)        : Numeric format of Sleep time of the user
    """

    morning_hours = config.get('general_config').get('wakeup_probable_hours')
    sleeping_hours = config.get('general_config').get('sleep_probable_hours')

    sleep_time_int = -1
    wakeup_time_int = -1

    if np.any(sleep_hours[morning_hours] == 1):
        wakeup_time = np.where(sleep_hours[morning_hours] == 1)[0][0] / samples_per_hour + morning_hours[0]/samples_per_hour
        wakeup_time_int = wakeup_time
    else:
        wakeup_time = -1

    if np.any(sleep_hours[sleeping_hours][::-1] == 1):
        sleep_time = ((len(sleeping_hours) - np.where(sleep_hours[sleeping_hours][::-1] == 1)[0][0]) /
                      samples_per_hour + sleeping_hours[0]/samples_per_hour) % Cgbdisagg.HRS_IN_DAY
        sleep_time_int = sleep_time
    else:
        sleep_time = -1

    if int(wakeup_time*samples_per_hour) not in config.get('general_config').get('wakeup_probable_hours'):
        wakeup_time = -1
        wakeup_time_int = -1
    else:
        fraction_time = wakeup_time - int(wakeup_time)
        fraction_time = np.digitize(fraction_time, [0, 0.25, 0.5, 0.75]) - 1
        fraction_time = ["00", "15", "30", "45"][fraction_time]
        wakeup_time = str(int(wakeup_time)) + ":" + fraction_time

    if int(sleep_time*samples_per_hour) not in config.get('general_config').get('sleep_probable_hours'):
        sleep_time = -1
        sleep_time_int = -1
    else:
        fraction_time = sleep_time - int(sleep_time)
        fraction_time = np.digitize(fraction_time, [0, 0.25, 0.5, 0.75]) - 1
        fraction_time = ["00", "15", "30", "45"][fraction_time]
        sleep_time = str(int(sleep_time)) + ":" + fraction_time

    return wakeup_time, sleep_time, sleep_time_int, wakeup_time_int
