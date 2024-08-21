"""
Author - Prasoon Patidar
Date - 03rd June 2020
Create new dictionary to store inputs required across all lifestyle submodules
"""

# import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.logs_utils import log_prefix
from python3.utils.time.get_time_diff import get_time_diff
from python3.analytics.lifestyle.init_lifestyle_config import init_lifestyle_config

from python3.analytics.lifestyle.functions.lifestyle_utils import get_vacation_percentage

from python3.initialisation.load_files.load_lifestyle_models import get_pilot_based_info
from python3.initialisation.load_files.load_lifestyle_models import get_daily_kmeans_lifestyle_models

from python3.analytics.lifestyle.functions.prepare_lifestyle_input_data import get_cooling_estimate
from python3.analytics.lifestyle.functions.prepare_lifestyle_input_data import prepare_lifestyle_input_data


def init_lifestyle_input_object(disagg_input_object, disagg_output_object, logger_pass):

    """
    Parameters:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
        logger_pass(dict)                       : Contains base logger and logging dictionary

    Returns:
        lifestyle_input_object(dict)            : Dictionary containing all inputs for lifestyle modules and submodules
    """

    t_init_lifestyle_input_object_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('initialize_lifestyle_input_object')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    # Initialize dict for lifestyle_input_object with user meta information

    global_config = disagg_input_object.get('config')

    lifestyle_input_object = {
        'uuid'         : global_config.get('uuid'),
        'pilot_id'     : global_config.get('pilot_id'),
        'sampling_rate': global_config.get('sampling_rate')
    }

    # Get gb disagg event start end times

    lifestyle_input_object.update({
        'event_start_time': disagg_input_object.get('out_bill_cycles_by_module').get('lifestyle_profile')[0][0],
        'event_end_time'  : disagg_input_object.get('out_bill_cycles_by_module').get('lifestyle_profile')[-1][-1],
        'disagg_mode'     : global_config.get("disagg_mode")
    })

    # get out bill cycles for this user run

    lifestyle_input_object.update({
        'out_bill_cycles': disagg_input_object.get('out_bill_cycles_by_module').get('lifestyle_profile')
    })

    # Initialize static config for lifestyle, and update lifestyle input object

    lifestyle_config = init_lifestyle_config()

    lifestyle_input_object.update(lifestyle_config)

    # prepare raw input data for lifestyle modules

    input_data_lf, day_input_data_lf, input_data_indices = prepare_lifestyle_input_data(lifestyle_input_object,
                                                                                        disagg_input_object,
                                                                                        disagg_output_object,
                                                                                        logger_pass)

    # add input and corresponding indices in lifestyle input object

    lifestyle_input_object['input_data'] = input_data_lf

    lifestyle_input_object['behavioural_profile'] = disagg_output_object.get('behavioural_profile')

    lifestyle_input_object['day_input_data'] = day_input_data_lf

    lifestyle_input_object.update(input_data_indices)

    # get hvac epoch level estimate in lifestyle input object

    cooling_epoch_estimate = get_cooling_estimate(lifestyle_input_object, disagg_input_object, disagg_output_object,
                                                  logger_pass)

    # write hvac estimates in lifestyle input object

    lifestyle_input_object['cooling_epoch_estimate'] = cooling_epoch_estimate

    # adding lifestyle HSM data

    lifestyle_input_object['lifestyle_hsm'] = copy.deepcopy(disagg_input_object.get('appliances_hsm').get('li'))

    if lifestyle_input_object.get('disagg_mode') == 'historical':
        lifestyle_input_object['lifestyle_hsm'] = None

    # get vacation percentage for each bill cycle

    bc_vacation_info, day_vacation_info = get_vacation_percentage(lifestyle_input_object, disagg_input_object, disagg_output_object)

    lifestyle_input_object['bc_vacation_info'] = bc_vacation_info

    lifestyle_input_object['day_vacation_info'] = day_vacation_info

    logger.info("%s Got vacation information for each bill cycle", log_prefix('VacationPercentage'))

    # get lighting bands for complete data using lighting hsm

    li_bands_empty = np.zeros(Cgbdisagg.HRS_IN_DAY)

    li_hsm_attributes = disagg_output_object.get('created_hsm', {}).get('li')

    li_band_hourly = li_bands_empty

    if li_hsm_attributes is not None:

        li_band_sampling_rate = li_hsm_attributes.get('attributes', {}).get('sleep_hours')

        samples_per_day = int(Cgbdisagg.SEC_IN_DAY / global_config.get('sampling_rate'))

        if (li_band_sampling_rate is not None) & (len(li_band_sampling_rate) == samples_per_day):
            samples_per_hour = int(Cgbdisagg.SEC_IN_HOUR / global_config.get('sampling_rate'))

            li_band_hourly = \
                np.array([np.max(li_band_sampling_rate[i:i + samples_per_hour])
                          for i in range(0, len(li_band_sampling_rate), samples_per_hour)])

            logger.debug("%s Got hourly lighting bands: %s",
                         log_prefix(['WakeUpTime','SleepTime'], type='list'), str(li_band_hourly.tolist()))

    if np.sum(li_band_hourly) <= 0:

        logger.debug("%s Unable to get hourly lighting bands: returning empty lighting bands",
                     log_prefix(['WakeUpTime','SleepTime'], type='list'))

    lifestyle_input_object['lighting_hourly_bands'] = li_band_hourly

    # Get pilot level info for this user's pilot

    pilot_id = lifestyle_input_object.get('pilot_id')
    pilot_id_str = str(pilot_id)

    pilot_based_info = get_pilot_based_info(disagg_input_object, pilot_id_str, logger_pass)

    lifestyle_input_object['pilot_based_config'] = pilot_based_info

    logger.debug("%s Got pilot level config for user from static files", log_prefix('DailyLoadType'))

    # Get model id for kmeans daily load model for user

    kmeans_model_id = get_model_id_for_user(lifestyle_input_object, logger_pass)

    logger.debug("%s Got Daily Kmeans Model Id: %s", log_prefix('DailyLoadType'), str(kmeans_model_id))

    # Get daily kmeans model corresponding to model id

    lifestyle_input_object['daily_profile_kmeans_model'] = \
        get_daily_kmeans_lifestyle_models(disagg_input_object,
                                          kmeans_model_id, pilot_id_str,
                                          logger_pass)

    # Get yearly kmeans model corresponding to model id

    all_models = disagg_input_object.get('loaded_files')
    all_lifestyle_models = all_models.get('lf_files')
    yearly_kmeans_model = all_lifestyle_models.get('yearly_kmeans_model')

    if pilot_id_str not in yearly_kmeans_model.keys():
        # Use global comparision keys to process, but throw warning

        pilot_based_info_key = 'universal'

        logger.warning("%s pilot id is not present in yearly kmeans model, using universal config for this user",
                       pilot_id_str)

    else:

        pilot_based_info_key = pilot_id_str

        logger.info("%s pilot id is present in yearly kmeans model, using pilot config for this user",
                    pilot_id_str)

    yearly_kmeans_model = yearly_kmeans_model.get(pilot_based_info_key)
    kmeans_model_id_key_mapping = {
        'c': 'consumption',
        'v': 'variation',
        'l': 'low',
        'a': 'avg',
        'h': 'high'
    }

    model_key = '_'.join([kmeans_model_id_key_mapping.get(k, '') for k in kmeans_model_id])

    model = yearly_kmeans_model.get(model_key)

    lifestyle_input_object['yearly_profile_kmeans_model'] = model

    logger.debug("%s Got kmeans models for daily and yearly load type for user from static files",
                 log_prefix(['DailyLoadType','SeasonalLoadType'], type='list'))

    t_init_lifestyle_input_object_end = datetime.now()

    logger.info("%s Initializing lifestyle input object took | %.3f s", log_prefix('Generic'),
                get_time_diff(t_init_lifestyle_input_object_start, t_init_lifestyle_input_object_end))

    return lifestyle_input_object


def get_model_id_for_user(lifestyle_input_object, logger_pass):

    """
    Parameters:
        lifestyle_input_object(dict)            : Dictionary containing all inputs for lifestyle modules
        logger_pass(dict)                       : Contains base logger and logging dictionary

    Returns:
        kmeans model id(str)                    : Daily load kmeans model id of user based on consumption attributes
    """

    t_get_model_id_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('initialize_lifestyle_input_object')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    # Get input data and pilot info from config

    input_data = lifestyle_input_object.get('input_data')

    pilot_based_info = lifestyle_input_object.get('pilot_based_config')

    kmeans_model_config = lifestyle_input_object.get('kmeans_model_config')

    # Get Daily Consumption values from input_data

    day_vals_unique, day_vals_inv_idx = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)

    daily_consumption_vals = np.bincount(day_vals_inv_idx, weights=input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # Get Median, IQD for daily consumption

    median_daily_consumption = np.median(daily_consumption_vals)

    iqd_daily_consumption = np.percentile(daily_consumption_vals, 75) - np.percentile(daily_consumption_vals, 25)

    # Find Consumption Bucket for user based on pilot config
    # TODO(Nisha): remove static names and shift it to lifestyle config

    pilot_median_buckets = pilot_based_info.get('median_daily_cons_quantile_vals')

    pilot_iqd_buckets = pilot_based_info.get('iqd_daily_cons_quantile_vals')

    # Get consumption bucket id for this user

    if median_daily_consumption > pilot_median_buckets[-1]:

        # If median consumption more than final bucket limit, set bucket to final bucket

        user_cons_bucket = len(pilot_median_buckets) - 1

    else:

        # Get bucket based on where median consumption of user falls

        user_cons_bucket = np.where(pilot_median_buckets > median_daily_consumption)[0][0]

    # Get variation bucket id for this user

    if iqd_daily_consumption > pilot_iqd_buckets[-1]:

        # If iqd daily consumption more than final bucket limit, set bucket to final bucket

        user_iqd_bucket = len(pilot_iqd_buckets) - 1

    else:

        # Get bucket based on where iqd consumption of user falls

        user_iqd_bucket = np.where(pilot_iqd_buckets > iqd_daily_consumption)[0][0]

    # Check if user falls into allowed buckets

    allowed_user_buckets = lifestyle_input_object.get('pilot_based_config').get('allowed_user_buckets')

    for i in range(len(allowed_user_buckets)):
        allowed_user_buckets[i] = (int(str(allowed_user_buckets[i])[0]), int(str(allowed_user_buckets[i])[1]))

    user_bucket = (user_cons_bucket, user_iqd_bucket)

    if user_bucket in allowed_user_buckets:

        logger.info("%s User Bucket %s present in allowed user buckets", log_prefix('DailyLoadType'), str(user_bucket))


    else:
        default_user_bucket = kmeans_model_config.get('default_user_bucket')

        logger.warning("%s User Bucket %s not present in allowed user buckets, using default bucket %s ",
                       log_prefix('DailyLoadType'), str(user_bucket), str(default_user_bucket))

        user_bucket = default_user_bucket

    # get model id based on user bucket

    model_id_map = lifestyle_input_object.get('pilot_based_config').get('model_id_map')

    daily_load_kmeans_model_id = model_id_map[allowed_user_buckets.index(user_bucket)]

    t_get_model_id_end = datetime.now()

    logger.info("%s Generating kmeans model id for user took | %.3f s", log_prefix('DailyLoadType'),
                get_time_diff(t_get_model_id_start, t_get_model_id_end))

    return daily_load_kmeans_model_id
