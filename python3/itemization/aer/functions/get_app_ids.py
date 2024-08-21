
"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Calculate appliance profile information
"""

# Import python packages

import copy

import numpy as np

from python3.itemization.aer.functions.get_config import get_hybrid_config


def app_id_conf(item_input_object):

    """
    get app if config

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
    Returns:
        config                    (dict)          : list of stat app ids
    """

    # Table containing app id, app category, app type

    config = np.array([ [5, 1, 1],
                        [22, 1, 1],
                        [23, 1, 2],
                        [24, 1, 3],
                        [25, 1, 3],
                        [26, 1, 3],
                        [36, 1, 3],
                        [37, 1, 3],
                        [38, 1, 3],
                        [39, 1, 3],
                        [40, 1, 3],
                        [83, 1, 3],
                        [6, 3, 3],
                        [30, 3, 1],
                        [33, 3, 1],
                        [59, 3, 1],
                        [92, 3, 1],
                        [28, 2, 1],
                        [29, 2, 2],
                        [60, 2, 3],
                        [61, 2, 3],
                        [64, 2, 3],
                        [65, 2, 3],
                        [66, 2, 2],
                        [85, 2, 3],
                        [86, 2, 3],
                        [31, 3, 2]])

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    if hybrid_config.get("dishwash_cat") == "cook":
        config[config[:, 0] == 31] = np.array([31, 1, 1])

    return np.array(config)


def app_id_count(item_input_object, app_profile, logger):

    """
    Calculate appliance profile information from given app profile

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        app_profile               (dict)      : App profile data
        logger                    (logger)    : logger object

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
    """

    # Initialize default app count

    default_cooking = np.array([1, 1, 3])
    default_ent = np.array([1, 1, 0])

    pilot_level_config = item_input_object.get('pilot_level_config')
    mid_cons = pilot_level_config.get('cook_config').get('bounds').get('mid_cons')
    default_cooking = default_cooking * [0.3, 0.6, 1, 1.2][np.digitize(mid_cons, [30, 80, 150])]

    mid_cons = pilot_level_config.get('ld_config').get('bounds').get('mid_cons')
    default_laundry = [[1, 0, 0], [1, 1, 0], [1, 1, 1]][np.digitize(mid_cons, [40, 150])]
    default_laundry = np.array(default_laundry)

    default_cooking_flag = 1
    default_laundry_flag = 1
    default_ent_flag = 1

    # Initialize default app type

    cook_type = np.ones(3)
    ent_type = np.ones(3)
    ld_type = np.ones(3)

    appliance_profile = {
        "cooking": default_cooking,
        "ent": default_ent,
        "laundry": default_laundry,
        "cooking_type": cook_type,
        "ent_type": ent_type,
        "laundry_type": ld_type,
        "default_cooking_flag": default_cooking_flag,
        "default_laundry_flag": default_laundry_flag,
        "default_ent_flag": default_ent_flag,
        "default_cooking_count": default_cooking,
        "default_laundry_count": default_laundry,
        "default_ent_count": default_ent,
        "drier_present": 1
    }

    item_input_object.update({
        "appliance_profile": appliance_profile
    })

    drop_app_ids = item_input_object.get('pilot_level_config').get('ld_config').get("drop_app")

    logger.info("App IDS to be dropped | %s", drop_app_ids)

    # Default statistical app profile

    non_empty_app_profile_flag = (app_profile is None) or (app_profile == {}) or (app_profile == []) or (not len(app_profile))

    if (non_empty_app_profile_flag):
        logger.info("Statistical appliances app profile absent")
        return item_input_object

    # prepare app prof count and feul type array for all subcategories of cooking/laundry/entertainment app profile
    # this array either contains default app count or app count based on app profile information

    app_data = app_id_conf(item_input_object)

    cook_count, cook_type, ent_count, ent_type, ld_count, ld_type, app_ids, app_count, app_type = \
        prepare_stat_app_count_and_type_arr(app_data, app_profile, drop_app_ids, default_cooking, default_ent, default_laundry)

    ent_type = np.array([1, 1, 1])
    cook_type[2] = 1

    ld_type = ld_type.astype(bool)
    ent_type = ent_type.astype(bool)

    default_cooking_flag = int(not np.any(cook_count != default_cooking))
    default_laundry_flag = int(not np.any(ld_count != default_laundry))
    default_ent_flag = int(not np.any(ent_count != default_ent))

    default_cooking_flag = default_cooking_flag and int(np.sum(cook_type) == 3)
    default_laundry_flag = default_laundry_flag and int(np.sum(ld_type) == 3)
    default_ent_flag = default_ent_flag and int(np.sum(ent_type) == 3)

    drier_present = 1

    # if drier app count is 0 or fuel type is gas, laundry app count is decreased

    drier_info_present_in_profile = 6 in app_ids and (not ((app_profile.get(6) is None) or non_empty_app_profile_flag))

    if drier_info_present_in_profile:
        default_laundry_flag = 0
        ld_count[2] = app_count[app_ids == 6][0]
        ld_type[2] = int(not (app_profile.get(6).get('type') == 'GAS'))

        drier_present = ld_count[2]

        ld_count[0] = max(0, ld_count[0] - 0.5 * ((ld_count[2] * ld_type[2]) == 0))
        ld_count[1] = max(0, ld_count[1] - 0.25 * ((ld_count[2] * ld_type[2]) == 0))

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    # if dishwasher app count is 0, cooking app count is decreased

    if not default_laundry_flag and hybrid_config.get("dishwash_cat") == "cook":
        ld_count[1] = 0

    # if washing machine app count is 0, laundry app count is decreased

    ld_count, default_laundry_flag = \
        update_stat_app_count(30, app_profile, non_empty_app_profile_flag, default_laundry_flag, app_count, app_ids,
                              ld_count, 0)


    # if laundry app count is 0, laundry app count is decreased

    ld_count, default_laundry_flag = \
        update_stat_app_count(59, app_profile, non_empty_app_profile_flag, default_laundry_flag, app_count, app_ids,
                              ld_count, 0)

    # if washer app count is 0, laundry app count is decreased

    ld_count, default_laundry_flag = \
        update_stat_app_count(92, app_profile, non_empty_app_profile_flag, default_laundry_flag, app_count, app_ids,
                              ld_count, 0)


    # if dishwasher app count is 0, and it belongs to laundry, laundry app count is decreased

    ld_count, default_laundry_flag = \
        update_stat_app_count_for_dishwasher(31, app_profile, non_empty_app_profile_flag, default_laundry_flag,
                                             app_count, app_ids, ld_count, 1, (not hybrid_config.get("dishwash_cat") == "cook"))

    # if dishwasher app count is 0, and it belongs to laundry, laundry app count is decreased

    ld_count, default_laundry_flag = \
        update_stat_app_count_for_dishwasher(33, app_profile, non_empty_app_profile_flag, default_laundry_flag,
                                             app_count, app_ids, ld_count, 1, (not hybrid_config.get("dishwash_cat") == "cook"))

    # if dishwasher app count is 0, and it belongs to cooking, cooking app count is decreased

    cook_count, default_cooking_flag = \
        update_stat_app_count_for_dishwasher(31, app_profile, non_empty_app_profile_flag, default_cooking_flag,
                                             app_count, app_ids, cook_count, 1, hybrid_config.get("dishwash_cat") == "cook")

    # if dishwasher app count is 0, and it belongs to cooking, cooking app count is decreased

    cook_count, default_cooking_flag = \
        update_stat_app_count_for_dishwasher(33, app_profile, non_empty_app_profile_flag, default_cooking_flag,
                                             app_count, app_ids, cook_count, 1, hybrid_config.get("dishwash_cat") == "cook")

    # if cooking app count is 0, cooking app count is decreased

    cook_count, default_cooking_flag = \
        update_stat_app_count(5, app_profile, non_empty_app_profile_flag, default_cooking_flag, app_count, app_ids,
                              cook_count, 0)

    # if entertainment app count is 0, entertainment app count is decreased

    ent_count, default_ent_flag = \
        update_stat_app_count(66, app_profile, non_empty_app_profile_flag, default_ent_flag, app_count, app_ids,
                              ent_count, 1)

    # if oven app count is 0, cooking app count is decreased

    cook_count, default_cooking_flag = \
        update_stat_app_count(23, app_profile, non_empty_app_profile_flag, default_cooking_flag, app_count, app_ids,
                              cook_count, 1)

    # if microwave app count is 0, cooking app count is decreased

    cook_count, default_cooking_flag = \
        update_stat_app_count(25, app_profile, non_empty_app_profile_flag, default_cooking_flag, app_count, app_ids,
                              cook_count, 2)

    # if cooking range app count is 0, cooking app count is decreased

    cook_count, default_cooking_flag = \
        update_stat_app_count(22, app_profile, non_empty_app_profile_flag, default_cooking_flag, app_count, app_ids,
                              cook_count, 0)

    cook_count, cook_type = update_app_count_for_low_cook(item_input_object, cook_count, cook_type)

    if np.sum(cook_count) == 0:
        cook_count[0] = 1

    if np.sum(ld_count) == 0:
        ld_count[0] = 0.5

    if np.sum(ent_count) == 0:
        ent_count[0] = 1

    # Update app profile

    appliance_profile = {
        "cooking": np.fmax(0, cook_count),
        "ent": np.fmax(0, ent_count),
        "laundry": np.fmax(0, ld_count),
        "cooking_type": cook_type,
        "ent_type": ent_type,
        "laundry_type": ld_type,
        "default_cooking_flag": default_cooking_flag,
        "default_laundry_flag": default_laundry_flag,
        "default_ent_flag": default_ent_flag,
        "default_cooking_count": default_cooking,
        "default_laundry_count": default_laundry,
        "default_ent_count": default_ent,
        "drier_present": drier_present
    }

    logger.info("Default cooking app profile flag %s", default_cooking_flag)
    logger.info("Default laundry app profile flag %s", default_laundry_flag)
    logger.info("Default entertainment app profile flag %s", default_ent_flag)
    logger.info("Cooking appliance count %s", cook_count)
    logger.info("Entertainment appliance count %s", ent_count)
    logger.info("Laundry appliance count flag %s", ld_count)
    logger.info("Cooking appliance app type %s", cook_type)
    logger.info("Laundry appliance app type %s", ld_type)

    item_input_object.update({
        "appliance_profile": appliance_profile
    })

    return item_input_object


def update_app_count_for_low_cook(item_input_object, cook_count, cook_type):

    """
      Update stat app count if a particular app count is 0

      Parameters:
          cook_count              (np.ndarray)      : app count for each stat app category
          cook_type               (np.ndarray)      : fuel type for each stat app category
      Returns:
          cook_count              (np.ndarray)      : app count for each stat app category
          cook_type               (np.ndarray)      : fuel type for each stat app category
      """

    default_cooking_fuel_is_electric = str(np.nan_to_num(item_input_object.get('pilot_level_config').get('cook_config').get('type'))) == 'ELECTRIC'
    default_cooking_fuel_is_gas = str(np.nan_to_num(item_input_object.get('pilot_level_config').get('cook_config').get('type'))) == 'GAS'

    if (((cook_count[0] * cook_count[1]) == 0) * (cook_count[1] + cook_count[0] > 0)) and (not default_cooking_fuel_is_gas):
        cook_count[2] = max(0, cook_count[2] - 1.5)
        cook_count[0] = max(0, cook_count[0] - 0.5)
        cook_count[1] = max(0, cook_count[1] - 0.5)

    elif (cook_count[0] + cook_count[1]) == 0:
        cook_count[2] = max(1, cook_count[2] - 2)

    elif (cook_type[0] + cook_type[1]) == 0:
        cook_count[2] = max(1, cook_count[2] - 2 - int(default_cooking_fuel_is_electric))

    if default_cooking_fuel_is_gas:
        cook_type[cook_type == 0] = 1

    return cook_count, cook_type


def update_stat_app_count(app_id, app_profile, non_empty_app_profile_flag, stat_app_type, app_count, app_id_list,
                          stat_app_count, category_idx):

    """
    Update stat app count if a particular app count is 0

    Parameters:
        app_id                      (int)             : app id of target appliance
        app_profile                 (dict)            : user app profile info
        non_empty_app_profile_flag  (bool)            : this variable is true if app profile is empty
        stat_app_type               (np.ndarray)      : fuel type for each stat app category
        app_count                   (np.ndarray)      : app count of all appliances
        app_id_list                 (np.ndarray)      : app id list of all appliances
        stat_app_count              (np.ndarray)      : app count for each stat app category
        category_idx                (int)             : stat app sub category index to be targeted for current app id
    Returns:
        stat_app_count              (np.ndarray)      : app count for each stat app category
        stat_app_type               (np.ndarray)      : fuel type for each stat app category
    """

    target_app_present_in_app_profile = app_id in app_id_list and (not ((app_profile.get(app_id) is None) or non_empty_app_profile_flag))

    if target_app_present_in_app_profile:
        stat_app_type = 0
        if app_count[app_id_list == app_id][0] == 0:
            stat_app_count[category_idx] = 0

    return stat_app_count, stat_app_type


def update_stat_app_count_for_dishwasher(app_id, app_profile, non_empty_app_profile_flag, stat_app_type,
                                         app_count, app_id_list, stat_app_count, category_idx, dishwashflag):

    """
    Update stat app count if a particular app count is 0 (for dishwasher app ids)

    Parameters:
        app_id                      (int)             : app id of target appliance
        app_profile                 (dict)            : user app profile info
        non_empty_app_profile_flag  (bool)            : this variable is true if app profile is empty
        stat_app_type               (np.ndarray)      : fuel type for each stat app category
        app_count                   (np.ndarray)      : app count of all appliances
        app_id_list                 (np.ndarray)      : app id list of all appliances
        stat_app_count              (np.ndarray)      : app count for each stat app category
        category_idx                (int)             : stat app sub category index to be targeted for current app id
        dishwashflag                (bool)            : flag to deno whether dishwasher belongs to this app category
    Returns:
        stat_app_count              (np.ndarray)      : app count for each stat app category
        stat_app_type               (np.ndarray)      : fuel type for each stat app category
    """

    target_app_present_in_app_profile = app_id in app_id_list and (not ((app_profile.get(app_id) is None) or non_empty_app_profile_flag))

    if target_app_present_in_app_profile:
        stat_app_type = 0
        if app_count[app_id_list == app_id][0] == 0 and dishwashflag:
            stat_app_count[category_idx] = 0

    return stat_app_count, stat_app_type


def fill_stat_app_count_and_type(profile_idx, stat_app_ind, app_count, app_type, stat_app_count, stat_app_type):

    """
    helper function for updating count and fuel type from given stat app category based on individual app count

    Parameters:
        profile_idx           (int)             : index of the target appliance
        stat_app_ind          (int)             : index of stat app category
        app_count             (np.ndarray)      : app count of the target appliance
        app_type              (np.ndarray)      : fuel type of the target appliance
        def_app_count         (np.ndarray)      : default app count for each appliance
        stat_app_count        (np.ndarray)      : app count for each stat app category
        stat_app_type         (np.ndarray)      : fuel type for each stat app category

    Returns:
        stat_app_count        (np.ndarray)      : app count for each stat app category
        stat_app_type         (np.ndarray)      : fuel type for each stat app category
    """

    if app_count[profile_idx] is not None:
        stat_app_count[stat_app_ind] = np.nan_to_num(stat_app_count[stat_app_ind]) + app_count[profile_idx]

    if app_type[profile_idx] is not None:
        stat_app_type[stat_app_ind] = np.nan_to_num(stat_app_type[stat_app_ind]) + int(app_type[profile_idx])

    return stat_app_count, stat_app_type


def fill_def_val(app_data, app_ids, def_app_count, app_index, stat_app_count, stat_app_type):

    """
    Calculate count and fuel type from given stat app category, based on presence of default appliance information

    Parameters:
        app_data              (np.ndarray)      : app id and app category for each stat appliance
        app_ids               (np.ndarray)      : app id for each appliance
        def_app_count         (np.ndarray)      : default app count for each appliance
        app_index             (int)             : app index for the target app category
        stat_app_count        (np.ndarray)      : app count for each stat app category
        stat_app_type         (np.ndarray)      : fuel type for each stat app category

    Returns:
        stat_app_count        (np.ndarray)      : app count for each stat app category
        stat_app_type         (np.ndarray)      : fuel type for each stat app category
    """

    app_ids_list = app_data[np.logical_and(app_data[:, 1] == 1, app_data[:, 2] == (app_index + 1)), 0]

    main_app_ids = [30, 5]

    update_app_count_with_default_values = not np.all(np.isin(app_ids_list, app_ids)) and \
                                           (not len(np.intersect1d(app_data[app_data[:, 2] == app_index+1, 0], main_app_ids)))

    if update_app_count_with_default_values:
        stat_app_count[app_index] = max(stat_app_count[app_index], def_app_count[app_index])

    update_app_count_with_default_values = np.all(np.logical_not(np.isin(app_ids_list, app_ids))) and \
                                           (not len(np.intersect1d(app_data[app_data[:, 2] == app_index, 0], main_app_ids)))

    if update_app_count_with_default_values:
        stat_app_type[app_index] = 1

    return stat_app_count, stat_app_type


def prepare_stat_app_count_and_type_arr(app_data, app_profile, drop_app_ids, default_cooking, default_ent, default_laundry):

    """
    Calculate count and fuel type from given stat app category

    Parameters:
        app_data              (np.ndarray)      : app id and app category for each stat appliance
        app_profile           (dict)            : user app profile info
        drop_app_ids          (list)            : list of app ids to be droped for given pilot

    Returns:
        cook_count            (np.ndarray)      : app count for each cooking subcategories
        cook_type             (np.ndarray)      : fuel type for each cooking subcategories
        ent_count             (np.ndarray)      : app count for each entertainment subcategories
        ent_type              (np.ndarray)      : fuel type for each entertainment subcategories
        ld_count              (np.ndarray)      : app count for each laundry subcategories
        ld_type               (np.ndarray)      : fuel type for each laundry subcategories
        app_ids               (np.ndarray)      : app id for appliances in app profile
        app_count             (np.ndarray)      : app count for appliances in app profile
        app_type              (np.ndarray)      : app fuel type for appliances in app profile

    """

    app_ids = np.zeros(len(app_profile) + len(drop_app_ids))
    app_count = np.zeros(len(app_profile) + len(drop_app_ids))
    app_type = np.array(['ELECTRIC'] * (len(app_profile) + len(drop_app_ids)))

    # Fetch app id, count and type of all appliances

    for index, app_dict in enumerate(app_profile):
        app_dict = app_profile[app_dict]
        app_ids[index] = app_dict.get('appID')
        app_count[index] = app_dict.get('number')
        app_type[index] = app_dict.get('type')

        app_type[index] = int(not (app_type[index] == 'GAS')) + int((app_type[index] == 'ELECTRIC'))

    for index, app_dict in enumerate(drop_app_ids):
        app_ids[index + len(app_profile)] = int(drop_app_ids[index])
        app_count[index + len(app_profile)] = 0
        app_type[index + len(app_profile)] = 1

    app_count = np.fmin(6, app_count)
    app_count = np.fmax(0, app_count)

    # Initialize appliance count and type

    cook_count = np.empty(len(np.unique(app_data[app_data[:, 1] == 1, 2])))
    ent_count = np.empty(len(np.unique(app_data[app_data[:, 1] == 2, 2])))
    ld_count = np.empty(3)

    cook_type = np.empty(len(np.unique(app_data[app_data[:, 1] == 1, 2])))
    ent_type = np.empty(len(np.unique(app_data[app_data[:, 1] == 2, 2])))
    ld_type = np.empty(3)

    cook_count[:] = np.nan
    cook_type[:] = np.nan
    ent_count[:] = np.nan
    ent_type[:] = np.nan
    ld_count[:] = np.nan
    ld_type[:] = np.nan

    # for each index calculate app count and type

    for j in range(len(app_ids)):

        index = np.where(app_data[:, 0] == app_ids[j])[0]

        if app_data[index, 1] == 1:
            index = index[0]
            cook_count, cook_type = fill_stat_app_count_and_type(j, app_data[index, 2] - 1, app_count, app_type, cook_count, copy.deepcopy(cook_type))

        elif app_data[index, 1] == 2:
            index = index[0]
            ent_count, ent_type = fill_stat_app_count_and_type(j, app_data[index, 2] - 1, app_count, app_type, ent_count, copy.deepcopy(ent_type))

        elif app_data[index, 1] == 3:
            index = index[0]
            ld_count, ld_type = fill_stat_app_count_and_type(j, app_data[index, 2] - 1, app_count, app_type, ld_count, copy.deepcopy(ld_type))

    # Assign default values in absence of appliance profile

    cook_type[np.isnan(cook_type)] = 1
    ent_type[np.isnan(ent_type)] = 1
    ld_type[np.isnan(ld_type)] = 1

    cook_count[np.isnan(cook_count)] = default_cooking[np.isnan(cook_count)]
    ent_count[np.isnan(ent_count)] = default_ent[np.isnan(ent_count)]
    ld_count[np.isnan(ld_count)] = default_laundry[np.isnan(ld_count)]

    # Check if all appliances are not given a particular kind, consider default values

    for index in range(len(cook_type)):
        cook_count, cook_type = fill_def_val(app_data[app_data[:, 1] == 1], app_ids, default_cooking, index, cook_count, cook_type)

    for index in range(len(ent_type)):
        ent_count, ent_type = fill_def_val(app_data[app_data[:, 1] == 2], app_ids, default_ent, index, ent_count, ent_type)

    for index in range(len(ld_type)):
        ld_count, ld_type = fill_def_val(app_data[app_data[:, 1] == 3], app_ids, default_laundry, index, ld_count, ld_type)

    return cook_count, cook_type, ent_count, ent_type, ld_count, ld_type, app_ids, app_count, app_type
