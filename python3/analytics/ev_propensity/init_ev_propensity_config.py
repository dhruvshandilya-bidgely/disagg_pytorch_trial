"""
Author - Paras Tehria
Date - 27 May 2021
Call the init_ev_propensity_config function to load the required parameters for the EV propensity algorithm
"""


def init_ev_propensity_config(global_config, analytics_input_object):
    """
    Initialize the parameters required for running ev propensity
    Parameters:
        global_config                    (dict)            : Dictionary containing global config
        analytics_input_object              (dict)         : Dictionary containing information about analytics input
    Returns:
        ev_detection_config              (dict)            : Dictionary containing all configuration variables for ev
    """

    # 'default_adult_popul_perc': default value for ev propensity feature adult_population_perc
    # 'default_education_perc': default value for ev propensity feature population_perc_higher_education,
    # 'default_income': default value for ev propensity feature median_household_income
    # 'model_features': features used in ev propensity model,
    # 'age_group_buckets': age-groups in three age buckets('youth', 'adult', 'senior'),
    #
    # 'solar_weight': Weight given to solar user in post processing,
    # 'dwelling_weight': Weight given to dwelling bool in post processing,
    # 'ownership_weight': Weight given to ownership bool in post processing

    ev_propensity_config = {
        'uuid': global_config.get('uuid'),
        'pilot': global_config.get('pilot_id'),
        'zipcode': analytics_input_object.get('home_meta_data', {}).get('zip'),

        'dwelling': analytics_input_object.get('home_meta_data', {}).get('dwelling'),
        'ownershipType': analytics_input_object.get('home_meta_data', {}).get('ownershipType'),

        'default_adult_popul_perc': 27,
        'default_education_perc': 25,
        'default_income': 45000,

        'model_features': ['office_goer_bool', 'active_user_bool', 'dailyload_baseload',
                           'dailyload_eve', 'dailyload_other', 'zip_num_station', 'adult_population_perc',
                           'population_perc_higher_education', 'median_household_income'],

        'age_group_buckets': {"Under 5": 'youth', "5-9": 'youth', "10-14": 'youth', "15-19": 'youth',
                              "20-24": 'youth', "25-29": 'youth', "30-34": 'adult', "35-39": 'adult',
                              "40-44": 'adult', "45-49": 'adult', "50-54": 'senior',
                              "55-59": 'senior', "60-64": 'senior', "65-69": 'senior', "70-74": 'senior',
                              "75-79": 'senior', "80-84": 'senior', "85 Plus": 'senior'},

        'solar_weight': 0.15,
        'dwelling_weight': 0.03,
        'ownership_weight': 0.03
    }

    return ev_propensity_config
