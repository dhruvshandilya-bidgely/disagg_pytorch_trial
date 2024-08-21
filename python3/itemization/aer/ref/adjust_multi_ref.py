
"""
Author - Nisha Agarwal
Date - 10/9/20
Adjust ref amplitude for users with multiple ref
"""


def adjust_multi_ref(app_id_count, ref_estimate, ref_config, logger):

    """
    Adjust ref amplitude for users with multiple ref

    Parameters:
        app_id_count            (list)             : Count of each kind of refrigerator
        ref_estimate            (int)              : Estimated ref day level consumption
        ref_config              (dict)             : Dictionary containing all information
        logger                  (logger)           : logger object

    Returns:
        ref_estimate            (int)   `          : Day level ref output
    """

    # Fetch columns of each

    ref_column = ref_config.get('app_profile').get("ref_column")
    compact_ref_column = ref_config.get('app_profile').get("compact_ref_column")
    freezer_column = ref_config.get('app_profile').get("freezer_column")

    ref_factor = ref_config.get('app_profile').get("ref_factor")
    compact_ref_factor = ref_config.get('app_profile').get("compact_ref_factor")
    freezer_factor = ref_config.get('app_profile').get("freezer_factor")

    # Increment factor for each appliance kind

    factor = [ref_factor, compact_ref_factor, freezer_factor]

    final_increment_factor = 0

    # If high number of ref are present, decrease the factor

    if app_id_count[ref_column] > ref_config.get("app_profile").get("ref_mid_limit"):
        factor[ref_column] = factor[ref_column] - (ref_config.get('app_profile').get("ref_limit") -
                                                   ref_config.get('app_profile').get("ref_mid_limit")) * \
                                                   ref_config.get("app_profile").get("ref_factor_decrement")

    # Subtract the ref i.e. already estimated, and then count total number of ref and increase the estimation

    app_id_count[app_id_count < 0] = 0

    if app_id_count[ref_column] > 0:

        logger.debug("Ref found in app profile | ")

        app_id_count[ref_column] = app_id_count[ref_column] - 1

        final_increment_factor = final_increment_factor + app_id_count[0] * factor[0]

    if app_id_count[compact_ref_column] > 0:

        logger.debug("Compact Ref found in app profile | ")

        final_increment_factor = final_increment_factor + app_id_count[1] * factor[1]

    if app_id_count[freezer_column] > 0:

        logger.debug("Freezer found in app profile | ")

        final_increment_factor = final_increment_factor + app_id_count[2] * factor[2]

    logger.info('Scaling factor for multi ref scenario | %s', (1 + final_increment_factor))

    ref_estimate = ref_estimate * (1 + final_increment_factor)

    return ref_estimate
