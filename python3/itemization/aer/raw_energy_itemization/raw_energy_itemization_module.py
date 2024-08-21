
"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Master file for itemization pipeline
"""

# Import python packages

import pandas as pd
from datetime import datetime

# import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.raw_energy_itemization.run_itemization_submodules import run_itemization_modules

from python3.itemization.aer.raw_energy_itemization.prepare_itemization_input_data import prepare_itemization_data


def run_raw_energy_itemization_modules(item_input_object, item_output_object, logger_pass):

    """
    Perform 100% itemization

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger_pass               (dict)      : logger dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    # Initialize the logger

    date_ts_list = item_output_object.get('debug').get("input_data_dict").get("date_ts_list")
    date_list = [datetime.utcfromtimestamp(ts) for ts in date_ts_list]
    date_list = pd.DatetimeIndex(date_list).date

    item_output_object.update({
        "date_list": date_list
    })

    t1 = datetime.now()

    # Prepare itemization input data

    item_input_object, item_output_object = prepare_itemization_data(item_input_object, item_output_object)

    # Running itemization modules

    item_input_object, item_output_object = run_itemization_modules(item_input_object, item_output_object, logger_pass)

    t3 = datetime.now()

    item_output_object["run_time"][1] = get_time_diff(t1, t3)

    return item_input_object, item_output_object
