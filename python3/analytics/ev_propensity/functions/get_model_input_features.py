"""
Author - Paras Tehria
Date - 27 May 2021
This is the main file that contains functions to generate input features for ev propensity model
"""

import logging
import traceback
import numpy as np

from uszipcode import SearchEngine


def get_adult_pop_perc(zipcode_data, debug, ev_propensity_config, logger):
    """
    This is the main function that generates input features for ev propensity model
    Parameters:
        zipcode_data                  (dict)              : ev propensity config
        debug                         (dict)              : Debug object for EV propensity
        ev_propensity_config           (dict)              : global config
        logger                        (logging.LoggerAdapter)           : logger object to generate logs
    Return:
        debug                         (dict)              : Debug object for EV propensity
    """

    age_group_buckets = ev_propensity_config.get("age_group_buckets")
    default_adult_popul_perc = ev_propensity_config.get("default_adult_popul_perc")

    age_bucket_dict = {}

    population_by_age_list_of_dicts = zipcode_data.get('population_by_age')

    # Get Population by various age groups
    if population_by_age_list_of_dicts is not None:
        for population_dict in population_by_age_list_of_dicts:

            # Three possible keys: male, female, total. We'll be using total population

            if population_dict.get('key') != "Total":
                continue

            age_group_list_of_dicts = population_dict.get('values')

            # Looping over age groups for bucketization
            for age_group_dict in age_group_list_of_dicts:
                age_group = age_group_dict.get('x')
                age_category = list(age_group_buckets.values())[age_group]
                age_bucket_dict[age_category] = age_bucket_dict.get(age_category, 0) + age_group_dict.get('y')

        adult_population_perc = 100 * age_bucket_dict.get('adult') / np.sum(list(age_bucket_dict.values()))

    else:
        logger.info("Population by age info not available for the zipcode, using default value | ")
        adult_population_perc = default_adult_popul_perc

    debug['adult_population_perc'] = adult_population_perc if adult_population_perc is not None else \
        default_adult_popul_perc

    return debug


def get_model_input_features(debug, ev_propensity_config, logger_base):
    """
    This is the main function that generates input features for ev propensity model
    Parameters:
        ev_propensity_config          (dict)              : ev propensity config
        debug                         (dict)             : Debug object for EV propensity
        logger_base                   (dict)             : logger object to generate logs
    Return:
        debug                         (dict)             : Debug object for EV propensity
    """

    # Initializing new logger child compute_ev_propensity

    logger_local = logger_base.get('logger_base').getChild('get_model_input_features')

    # logger

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # ------- zip-code level features ---------------

    zipcode = str(ev_propensity_config.get('zipcode'))
    default_adult_popul_perc = ev_propensity_config.get("default_adult_popul_perc")
    default_education_perc = ev_propensity_config.get("default_education_perc")
    default_income = ev_propensity_config.get("default_income")

    # noinspection PyBroadException
    try:
        search = SearchEngine(simple_or_comprehensive=SearchEngine.SimpleOrComprehensiveArgEnum.comprehensive,
                              db_file_path=debug.get('zipcode_db_file_path'))
        zipcode_data = search.by_zipcode(zipcode)
        zipcode_data = zipcode_data.to_dict()

        median_household_income = zipcode_data.get('median_household_income')
        debug['median_household_income'] = median_household_income if median_household_income is not None else default_income

        education_dict = zipcode_data.get('educational_attainment_for_population_25_and_over')

        total_users_with_educ = 0
        if education_dict is not None:

            # 0th index is for total
            education_dict = education_dict[0]

            # count of users in each education bucket
            for education_level_dict in education_dict.get('values'):
                debug[education_level_dict.get('x')] = education_level_dict.get('y')
                total_users_with_educ += education_level_dict.get('y')

            # Percentage of users in each education bucket

            for education_level_dict in education_dict.get('values'):
                debug[education_level_dict.get('x')] /= float(total_users_with_educ / 100)

            debug['population_perc_higher_education'] = debug.get("Bachelor's Degree") + debug.get("Master's Degree") + \
                                                        debug.get("Professional School Degree") + debug.get("Doctorate Degree")

        else:
            logger.info("Education info not available for the zipcode, using default value | ")
            debug['population_perc_higher_education'] = default_education_perc

        debug = get_adult_pop_perc(zipcode_data, debug, ev_propensity_config, logger)

    except Exception:

        # Imputing by median value
        error_str = (traceback.format_exc()).replace('\n', ' ')

        logger.warning("Warning : Error occurred in zipcode data computation, using default values. Error string | {}"
                       "".format(error_str))

        debug['population_perc_higher_education'] = default_education_perc
        debug['adult_population_perc'] = default_adult_popul_perc
        debug['median_household_income'] = default_income

    ev_station_df = debug.get('na_charging_station_data')

    if ev_station_df is not None and len(ev_station_df) > 0:
        debug['zip_num_station'] = np.count_nonzero(ev_station_df.ZIP.astype(str) == zipcode)
    else:
        logger.info("EV charging stations data not available, using num of station value as zero | ")
        debug['zip_num_station'] = 0

    # ------- User level features ---------------

    lifestyle_data = debug.get('lifestyle_attributes')

    debug['dailyload_baseload'] = float(lifestyle_data.get('DailyLoadType') == "BASE_LOAD")
    debug['dailyload_eve'] = float(
        lifestyle_data.get('DailyLoadType') in ["MORNING_EVE", "EARLY_EVE", "LATE_EVE"])
    debug['dailyload_other'] = float(not debug.get('dailyload_baseload') and not debug.get('dailyload_eve'))

    debug['office_goer_bool'] = float(lifestyle_data.get('OfficeGoer') == True)
    debug['active_user_bool'] = float(lifestyle_data.get('ActiveUser') == True)

    # List containing features in the order required by the model
    feature_ls = []
    for feature in ev_propensity_config.get('model_features'):
        logger.info("Value of EV propensity feature : {} | {}".format(feature, debug.get(feature, np.nan)))
        feature_ls.append(debug.get(feature, np.nan))

    return debug, feature_ls
