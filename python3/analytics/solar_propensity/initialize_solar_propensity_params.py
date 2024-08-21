"""
Author - Paras Tehria
Date - 17-Nov-2020
This module is used to initialize the parameters required for running solar propensity

"""

from python3.config.Cgbdisagg import Cgbdisagg


def init_solar_propensity_config(global_config, analytics_input_object):
    """
    Initialize the parameters required for running solar propensity

    Parameters:
        global_config       (dict)              : Dictionary containing global config
        analytics_input_object (dict)              : Dictionary containing information about disaggregation run

    Returns:
        solar_detection_config           (dict)              : Dictionary containing all configuration variables for solar
    """

    # 'uuid': UUID
    # 'pilot': Pilot ID
    # 'timezone': Timezone of the user, used in getting irradiance
    # 'longitude': Longitude,  used in getting irradiance
    # 'latitude': Latitude,  used in getting irradiance
    # 'dwelling': Dwelling Type
    # 'ownershipType': Ownership Type
    #
    #  'max_perc_neg_data':  Max allowed negative as fraction of total data length
    # 'rate_plan': Rate plan of the user
    # 'reverse_rate_plan': Rate plan at which utility buys back excess generation
    # 'panel_size_safety_factor': Safety factor used while calculating optimal panel size
    #
    # 'panel_price_dict':  Panel price according to the recommended panel size
    # 'max_consumption_quantile': Consumption capping %ile threshold
    # 'max_irradiance_quantile': Irradiance capping %ile threshold
    # 'irradiance_arr_col_dict': Irradiance arr column definition
    # 'sun_presence_col_idx': Column index of newly added sunlight presence column

    solar_propensity_config = {
        'uuid': global_config.get('uuid'),
        'pilot': global_config.get('pilot_id'),
        'timezone': analytics_input_object.get('home_meta_data').get('timezone'),
        'longitude': analytics_input_object.get('home_meta_data').get('longitude'),
        'latitude': analytics_input_object.get('home_meta_data').get('latitude'),
        'dwelling': analytics_input_object.get('home_meta_data').get('dwelling'),
        'ownershipType': analytics_input_object.get('home_meta_data').get('ownershipType'),

        'max_perc_neg_data': 0.05,

        'rate_plan': 0.1,
        'reverse_rate_plan': 0.075,
        'panel_size_safety_factor': 1.2,

        'panel_price_dict': {
            '3000': 6400,
            '4000': 8500,
            '5000': 10600,
            '6000': 12700,
            '7000': 14815,
            '8000': 17000,
            '9000': 19100,
            '10000': 21200
        },
        'max_consumption_quantile': 0.98,
        'max_irradiance_quantile': 0.98,
        'irradiance_arr_col_dict': {
            'epoch': 0,
            'irradiance': 1
        },

        'sun_presence_col_idx': Cgbdisagg.INPUT_DIMENSION,


    }

    return solar_propensity_config
