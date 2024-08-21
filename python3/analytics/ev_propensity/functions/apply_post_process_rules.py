"""
Author - Paras Tehria
Date - 27-May-2021
This module is used to inflate ev propensity
"""
# Import python packages
import logging


def apply_post_process_rules(ev_propensity_score, debug, ev_propensity_config, logger):
    """
    This function is used to inflate ev propensity value based on a few features

    Parameters:
        ev_propensity_score           (float)            : ev propensity score from the model
        ev_propensity_config          (dict)             : global config
        debug                         (dict)             : Debug object for EV propensity
        logger                        (logging.LoggerAdapter)   : logger object to generate logs
    Return:
        ev_propensity_score           (float)            : Modified ev propensity score from the model
    """

    solar_bool = int(debug.get("solar_present") == 1)

    logger.info("Solar bool | {}".format(solar_bool))

    dwelling_bool = int(debug.get("dwelling") == 1)
    logger.info("Dwelling Type | {}".format(debug.get("dwelling")))

    ownership_bool = int(debug.get('ownershiptype') == "Owned")
    logger.info("Ownership Type | {}".format(debug.get('ownershiptype')))

    solar_weight = ev_propensity_config.get('solar_weight')
    dwelling_weight = ev_propensity_config.get('dwelling_weight')
    ownership_weight = ev_propensity_config.get('ownership_weight')

    # EV propensity score is inflated if favorable value of above three features encountered
    ev_propensity_score += solar_weight * solar_bool + dwelling_weight * dwelling_bool + ownership_weight * ownership_bool

    return ev_propensity_score
