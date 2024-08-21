"""
Author - Mayank Sharan
Date - 27/01/19
This function calls all operations we need to do before we run the disagg
"""

# Import python packages

import logging

# Import functions from within the project

from python3.analytics.functions.disable_lifestyle import disable_lifestyle
from python3.analytics.functions.disable_ev_propensity import disable_ev_propensity
from python3.disaggregation.pre_disagg_ops.disable_pipeline import disable_pipeline
from python3.analytics.functions.disable_solar_propensity import disable_solar_propensity


def pre_analytics_ops(analytics_input_object):

    """
    Parameters:
        analytics_input_object (dict)              : Contains all inputs required to run the pipeline

    Returns:
        analytics_input_object (dict)              : Contains all inputs required to run the pipeline
    """

    # Initialize the logger

    logger_base = analytics_input_object.get('logger').getChild('pre_analytics_ops')
    logger = logging.LoggerAdapter(logger_base, analytics_input_object.get('logging_dict'))

    # Disable appliances for specific pilots and sampling rates

    analytics_input_object = disable_lifestyle(analytics_input_object, logger)
    analytics_input_object = disable_ev_propensity(analytics_input_object, logger)
    analytics_input_object = disable_solar_propensity(analytics_input_object, logger)
    analytics_input_object = disable_pipeline(analytics_input_object, logger)

    return analytics_input_object
