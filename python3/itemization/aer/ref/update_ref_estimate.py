"""
Author - Nisha Agarwal
Date - 10/9/20
Update the ref estimated amplitude based on information available from appliance profile
"""

# import python packages

import numpy as np

# import functions from within the project

from python3.itemization.aer.ref.adjust_multi_ref import adjust_multi_ref


def update_ref_estimate(ref_estimate, app_id_count, ref_config, logger):

    """
    Update the ref estimated amplitude based on information available from appliance profile

    Parameters:
        ref_estimate            (int)              : Estimated day level ref output
        app_id_count            (list)             : List of appliance counts
        ref_config              (dict)             : Dictionary containing all information
        logger                  (logger)           : logger object

    Returns:
        ref_estimate            (int)   `          : Modified day level ref output
    """

    ref_column = ref_config.get('app_profile').get("ref_column")
    freezer_column = ref_config.get('app_profile').get("freezer_column")
    compact_ref_column = ref_config.get('app_profile').get("compact_ref_column")

    ref_count_limit = ref_config.get('app_profile').get("ref_count_limit")
    freezer_count_limit = ref_config.get('app_profile').get("freezer_count_limit")
    compact_ref_count_limit = ref_config.get('app_profile').get("compact_ref_count_limit")

    # Safety check for number of appliances

    app_id_count[ref_column] = np.minimum(app_id_count[ref_column], ref_count_limit)
    app_id_count[freezer_column] = np.minimum(app_id_count[freezer_column], freezer_count_limit)
    app_id_count[compact_ref_column] = np.minimum(app_id_count[compact_ref_column], compact_ref_count_limit)

    if np.any(app_id_count >= 0):
        if np.sum(app_id_count > 0)==0 or app_id_count[0] == 0:
            logger.info("Adjusting ref consumption for single ref scenario")
            ref_estimate = 0
        elif app_id_count[0] == 1 and (app_id_count[1:] > 0).sum() == 0:
            logger.info("Adjusting ref consumption for single ref scenario")
        else:
            logger.info("Adjusting ref consumption for multi ref scenario")
            ref_estimate = adjust_multi_ref(app_id_count, ref_estimate, ref_config, logger)

    return ref_estimate
