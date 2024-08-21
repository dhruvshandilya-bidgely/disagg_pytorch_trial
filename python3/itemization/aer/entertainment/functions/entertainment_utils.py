
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Util function for entertainment estimation
"""

# Import python packages

import numpy as np
from numpy.random import RandomState

np.random.seed(1234)

from python3.itemization.init_itemization_config import random_gen_config


def get_start_end_time(start, end, usage_hours):

    """
       Calculate start and end time of entertainment appliance usage

       Parameters:
           start                    (float)              : Possible start time of appliance
           end                      (float)              : Possible end time of the appliance
           usage_hours              (int)                : usage hours of the appliance

       Returns:
           start_time               (float)              : Calculated start time of appliance
           end_time                 (float)              : Calculated end time of the appliance
       """

    seed = RandomState(random_gen_config.seed_value)

    hours_length = end - start

    # If given start and end time duration is greater than required duration, start time is randomly chosen

    if hours_length <= usage_hours:
        return start, end

    else:
        tou_list = np.arange(start, end - usage_hours)
        start_time = seed.choice(tou_list)
        end_time = start_time + usage_hours

        return start_time, end_time
