
"""
Author - Nisha Agarwal
Date - 10/9/20
Adjust ref amplitude for high consumption users
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def adjust_ref_estimate(day_estimate, item_input_object, ref_config, logger):

    """
    Adjust ref amplitude for high consumption users

    Parameters:
        day_estimate            (int)              : Estimated ref day level consumption
        item_input_object       (dict)             : Dictionary containing all inputs
        ref_config              (dict)             : Dictionary containing all information
        logger                  (logger)           : logger object

    Returns:
        day_estimate            (int)   `          : day level ref output
    """

    input_data = item_input_object.get("input_data")
    input_data = np.fmax(input_data, 0)

    # Number of target days of the user

    num_of_days = len(item_input_object.get("item_input_params").get("day_input_data"))

    # Calculate day level total consumption of the user

    user_day_level_energy = np.sum(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]) / num_of_days

    factor = ref_config.get("high_consumption").get("factors")

    # If the consumption level is greater than a given threshold, the day estimate is increased by a certain factor

    update_ref_cons_based_on_user_cons_level = user_day_level_energy > ref_config.get("high_consumption").get("limit") and (day_estimate < 1400)

    if update_ref_cons_based_on_user_cons_level:

        logger.debug("User day energy greater than the limit")

        # Fetch the required factor based on consumption bucket of the user

        consumption_bucket = np.digitize([user_day_level_energy], ref_config.get("high_consumption").get("energy_ranges"))

        day_estimate = day_estimate * (1 + factor[consumption_bucket[0]])

    return day_estimate
