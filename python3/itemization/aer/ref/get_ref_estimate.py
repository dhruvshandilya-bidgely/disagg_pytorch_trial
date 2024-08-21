"""
Author - Nisha Agarwal
Date - 10/9/20
Main script for estimating ref consumption
"""

# import functions from within the project

from python3.itemization.aer.ref.prepare_features import prepare_features
from python3.itemization.aer.ref.adjust_ref_estimate import adjust_ref_estimate
from python3.itemization.aer.ref.calculate_ref_estimate import calculate_ref_estimate


def get_ref_estimate(item_input_object, ref_config, logger):

    """
    Call all necessary functions to calculate day level estimate

    Parameters:
        item_input_object           (dict)             : Dictionary containing all inputs
        ref_config                    (dict)             : Dictionary containing all information
        logger                        (logger)           : logger object

    Returns:
        day_estimates                 (int)              : Day level ref estimate
        features                      (numpy.ndarray)    : Features used to estimate ref
    """

    # Prepare features for ref estimation

    features, model_category = prepare_features(item_input_object, ref_config, logger)

    logger.debug("Calculated ref features")

    # Calculate ref day level consumption and map all values at epoch level

    day_estimate = calculate_ref_estimate(item_input_object, features, model_category, logger)

    max_day_level_ref = 2000

    day_estimate = min(day_estimate, max_day_level_ref)

    logger.debug("Estimated ref consumption")

    # Adjust ref consumption for high consumption users or based on available meta features

    day_estimate = adjust_ref_estimate(day_estimate, item_input_object, ref_config, logger)

    logger.debug("Modified ref consumption based on app profile data")

    return day_estimate, features
