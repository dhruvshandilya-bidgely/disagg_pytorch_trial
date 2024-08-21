
"""
Author - Nisha Agarwal
Date - 10/9/20
Ref config file
"""

# No imports


def init_ref_config():

    """
    Initialize the config dictionary for ref module

    Returns:
        ref_config          (dict)              : Dictionary containing all config information
    """

    ref_config = dict()

    # Config required for estimation adjustment using appliance profile information

    # "ref_column": column for ref app count in app profile array
    # "compact_ref_column": column for compact ref app count in app profile array
    # "freezer_column": column for freezer app count in app profile array
    #
    # "ref_count_limit": max ref app count
    # "compact_ref_count_limit": max compact ref app count
    # "freezer_count_limit": max freezer app count
    #
    # "ref_app_ids":  ref app ids
    # "ref_app_ids_count": total ref app categories
    #
    # "ref_ids_columns": mapping of app profile with ref app category
    #
    # "ref_factor": ref multiplier
    # "compact_ref_factor": compact ref multiplier
    # "freezer_factor": freezer multiplier
    #
    # "ref_mid_limit": if total ref app profile greater than this values, modify ref app count,
    # "ref_factor_decrement": ref app count decrement factor ,
    #
    # "freezer_factor_decrement": freezer app count decrement factor ,

    ref_config['app_profile'] = {

        "ref_column": 0,
        "compact_ref_column": 1,
        "freezer_column": 2,

        "ref_count_limit": 5,
        "compact_ref_count_limit": 2,
        "freezer_count_limit": 2,

        "ref_app_ids": [9, 10, 58],
        "ref_app_ids_count": 3,

        "ref_ids_columns": [0, 1, 2],

        "ref_factor": 0.5,
        "compact_ref_factor": 0.4,
        "freezer_factor": 0.2,

        "ref_mid_limit": 3,
        "ref_factor_decrement": 0.1,
        "ref_limit": 1,

        "freezer_factor_decrement": 0.7

    }

    # "living_area_limits": living area min/max bound ,
    # "num_of_bedrooms_limits": bedrooms count min/max bound
    # "num_of_occupants_limits": occupants count min/max bound

    # Upper and lower limit for meta features

    ref_config['meta_data'] = {

        "living_area_limits": [400, 20000],
        "num_of_bedrooms_limits": [1, 12],
        "num_of_occupants_limits": [1, 10]
    }

    # Path to estimation model files

    ref_config["models"] = {
        "path": "/Users/nisha/PycharmProjects/slytherin/python3/itemization/ref/"
    }

    # Safety checks rules for final estimated ref amplitude

    # "min_limit": min day level ref ,
    # "max_limit": max day level ref,
    # "bedrooms_limit": higher ref for higher bedroom count,
    # "occupants_limit": higher ref for higher occupants count,
    # "meta_data_limit": min limit for higher expected ref consumption

    ref_config["limits"] = {
        "min_limit": 400,
        "max_limit": 3000,
        "bedrooms_limit": 4,
        "occupants_limit": 5,
        "meta_data_limit": 1000
    }

    # "limit": increase ref consumption for high consumption homes,
    # "factors": increment factors buckets,
    # "energy_ranges": energy ranges buckets

    # Config required for estimation adjustment using consumption level information

    ref_config["high_consumption"] = {
        "limit": 30000,
        "factors": [0, 0.1, 0.2, 0.2, 0.25, 0.3, 0.3, 0.3, 0.3, 0.3],
        "energy_ranges": [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    }

    # "percentile": Filter out low cons points in cooling consumption,
    # "hour_window": Filter out shorter windows in cooling consumption,
    # "day_window": Filter out shorter days usage in cooling consumption,

    # Config required for filtering hour level consumption

    ref_config["filter_consumption"] = {
        "percentile": 30,
        "hour_window": 4,
        "day_window": 8
    }

    # Config required for calculation of features

    ref_config["features"] = {
        "cooling_percentile": 97
    }

    return ref_config
