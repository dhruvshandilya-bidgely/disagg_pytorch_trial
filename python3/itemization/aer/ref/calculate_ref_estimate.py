
"""
Author - Nisha Agarwal
Date - 10/9/20
Calculate ref amplitude using respective model files
"""

# Import python packages

import numpy as np


def calculate_ref_estimate(item_input_object, features, model_category, logger):

    """
    Calculate ref amplitude using respective model files

    Parameters:
        item_input_object     (dict)             : Contains all hybrid  inputs
        features                (numpy.ndarray)    : Calculated features for ref estimation
        model_category          (str)              : category of model based on available features
        logger                  (logger)           : logger object

    Returns:
        day_estimate            (int)   `          : Estimted day level ref output
    """

    model_files = item_input_object.get("loaded_files").get("ref_files")

    day_estimate = 0

    features = np.array(features).reshape((1, -1))

    # estimate ref using only raw features

    if model_category == "raw":

        logger.info("Features used | Raw energy")

        model = model_files.get("raw")

        day_estimate = model.predict(features)[0]

    # estimate ref using raw and user meta features

    if model_category == "raw_meta":

        logger.info("Features used | Raw energy + HVAC")

        model = model_files.get("raw_meta")

        day_estimate = model.predict(features)[0]

    # estimate ref using  raw and cooling features

    if model_category == "raw_hvac":

        logger.info("Features used | Raw energy + HVAC")

        model = model_files.get("raw_hvac")

        day_estimate = model.predict(features)[0]

    # estimate ref using raw, HVAC and meta features

    if model_category == "raw_hvac_meta":

        logger.info("Features used | Raw energy + HVAC + meta")

        model = model_files.get("raw_hvac_meta")

        day_estimate = model.predict(features)[0]

    return day_estimate
